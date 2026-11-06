"""Qwen3.5-0.8B / 9B smoke, strict teacher-forced regression, and measurement."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from _common import ROOT, require_gpu, model_identity, fingerprint, write_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("smoke", "validate", "benchmark"), default="smoke")
    parser.add_argument("--backend", choices=("flash_attn", "triton_base", "triton_splitk", "triton_tc"), default="flash_attn")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", help="baseline validation output directory")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.tokens, args.batch, args.input_tokens, args.repeats) <= 0:
        parser.error("token, batch and repeat counts must be positive")
    if args.task == "validate" and args.reference is None and (args.backend, args.mode) != ("flash_attn", "eager"):
        parser.error("record a flash_attn eager baseline first")
    torch = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from paged_decode.hybrid import HybridRunner
    identity = model_identity(args.model)
    identity = {k: identity[k] for k in ("repo_id", "revision", "weights", "files")}
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "environment.json", fingerprint())
    write_json(destination / "arguments.json", {**vars(args), "model": identity})
    batch = args.batch if args.task == "benchmark" else 1
    maximum = max(1024, args.input_tokens + args.tokens)
    try:
        runner = HybridRunner(str(Path(args.model).resolve()), backend=args.backend, mode=args.mode,
                              batch=batch, max_context=maximum)
        write_json(destination / "runtime.json", runner.description())
        prompts = json.loads((ROOT / "configs/prompts.json").read_text(encoding="utf-8"))

        def encode(prompt):
            return runner.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                       tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)

        if args.task == "smoke":
            result = runner.generate([encode(prompts[0])], args.tokens)
            write_json(destination / "summary.json", {"status": "completed", "result": result})
            print(result["text"][0], flush=True)
        elif args.task == "validate":
            reports = []
            for index, prompt in enumerate(prompts):
                tokens = encode(prompt)
                prompt_hash = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                baseline = None
                if args.reference:
                    baseline = torch.load(Path(args.reference) / f"prompt-{index}.pt", weights_only=True, map_location="cpu")
                    if baseline["model"] != identity or baseline["prompt_sha256"] != prompt_hash:
                        raise ValueError("model or prompt differs from baseline")
                    if len(baseline["tokens"]) != args.tokens:
                        raise ValueError("baseline length differs")
                result = runner.generate([tokens], args.tokens, observe=True,
                         forced=[[t] for t in baseline["tokens"]] if baseline else None)
                if baseline is None:
                    torch.save({"model": identity, "prompt_sha256": prompt_hash,
                                "tokens": result["token_ids"][0], "logits": result["logits"]},
                               destination / f"prompt-{index}.pt")
                    reports.append({"prompt": index, "status": "recorded", "tokens": result["token_ids"][0]})
                    continue
                steps = []
                predicted = result["predicted_ids"][0]
                for step, (actual, expected) in enumerate(zip(result["logits"], baseline["logits"])):
                    actual, expected = actual.float(), expected.float()
                    delta = (actual - expected).abs()
                    top2 = expected[0].topk(2).values
                    steps.append({"step": step, "max_abs_error": delta.max().item(),
                                  "mse": delta.square().mean().item(),
                                  "logits_close": bool(torch.isfinite(actual).all() and
                                                       torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)),
                                  "baseline_token": baseline["tokens"][step], "candidate_token": predicted[step],
                                  "baseline_top2_gap": (top2[0] - top2[1]).item()})
                divergence = next((i for i, (a, b) in enumerate(zip(predicted, baseline["tokens"])) if a != b), None)
                report = {"prompt": index, "steps": steps, "first_argmax_difference": divergence,
                          "passed": divergence is None and all(s["logits_close"] for s in steps)}
                if divergence is not None:
                    report["free_run_tokens"] = runner.generate([tokens], args.tokens)["token_ids"][0]
                    torch.save({"step": divergence, "actual": result["logits"][divergence],
                                "expected": baseline["logits"][divergence]}, destination / f"divergence-{index}.pt")
                write_json(destination / f"prompt-{index}.json", report)
                reports.append(report)
            passed = all(r.get("passed", True) for r in reports)
            write_json(destination / "summary.json", {"status": "passed" if passed else "numerical_mismatch",
                       "model": identity, "backend": args.backend, "mode": args.mode, "prompts": reports})
            if not passed:
                raise SystemExit(2)
        else:
            generator = torch.Generator(device="cpu").manual_seed(927)
            prompts = torch.randint(100, 30000, (batch, args.input_tokens),
                                    generator=generator, device="cpu").tolist()
            prompt_hash = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
            runner.generate(prompts, args.tokens)
            rows = []
            for repeat in range(args.repeats):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = runner.generate(prompts, args.tokens, timing=True)
                elapsed = time.perf_counter() - started
                row = {"repeat": repeat, "backend": args.backend, "mode": args.mode,
                       "prompt_sha256": prompt_hash, "output_tokens": batch * args.tokens,
                       "pages_per_full_layer": batch * runner.cache.width,
                       "generation_wall_seconds": elapsed, "output_tokens_per_second": batch * args.tokens / elapsed,
                       "decode_model_ms_raw": result["decode_ms"], "decode_model_ms": summarize(result["decode_ms"]),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "token_ids": result["token_ids"],
                       "note": "fixed unpadded batch, upstream linear layers; decode forward includes KV writes/linear state/lm_head"}
                rows.append(row)
                write_json(destination / f"repeat-{repeat}.json", row)
            write_json(destination / "summary.json", {"status": "completed", "model": identity, "rows": rows,
                       "throughput": summarize([r["output_tokens_per_second"] for r in rows])})
    except Exception as exc:
        write_json(destination / "failure.json", {"status": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
