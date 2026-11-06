"""Record baseline or compare per-step logits under baseline-token teacher forcing.

Forward hooks are confined to this untimed diagnostic. Exact greedy token agreement
at every forced step implies identical free-run greedy output by induction. A failed
prompt additionally runs freely and records its first divergent token.
"""
import argparse
import hashlib
import json
from pathlib import Path
from _common import ROOT, require_gpu, model_identity, write_json, fingerprint


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("flash_attn", "triton_base", "triton_splitk"), default="flash_attn")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", help="baseline directory; absent means record baseline")
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.reference is None and (args.backend != "flash_attn" or args.mode != "eager"):
        parser.error("baseline must be flash_attn eager")
    torch = require_gpu()
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.block_manager import BlockManager
    identity = model_identity(args.model)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    prompts = json.loads((ROOT / "configs/prompts.json").read_text(encoding="utf-8"))
    identity_key = {key: identity[key] for key in ("repo_id", "revision", "files", "weights")}
    write_json(destination / "environment.json", fingerprint())
    llm = None
    reports = []
    try:
        llm = LLM(str(Path(args.model).resolve()), max_num_seqs=8, max_model_len=1024,
                  max_num_batched_tokens=4096, kv_cache_blocks=64,
                  decode_backend=args.backend, enforce_eager=args.mode == "eager",
                  gpu_memory_utilization=.75)
        sp = SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True)
        for index, prompt in enumerate(prompts):
            tokenized = llm.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                         tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if len(tokenized) + args.tokens > 1024:
                raise ValueError("prompt/output exceeds validation context limit")
            prompt_key = hashlib.sha256(json.dumps(tokenized).encode()).hexdigest()
            baseline = None
            if args.reference:
                baseline = torch.load(Path(args.reference) / f"prompt-{index}.pt", weights_only=True, map_location="cpu")
                if baseline["model"] != identity_key or baseline["prompt_sha256"] != prompt_key:
                    raise ValueError("baseline model or tokenized prompt identity differs")
                if len(baseline["tokens"]) != args.tokens:
                    raise ValueError("baseline output length differs")
            llm.scheduler.block_manager = BlockManager(64, 256)
            logits = []
            predicted = []
            cursor = [0]

            def observe(module, inputs, output):
                saved = output.detach().cpu().clone()
                logits.append(saved)
                predicted.append(int(saved[0].argmax()))

            def force_token(module, inputs, output):
                token = baseline["tokens"][cursor[0]]
                cursor[0] += 1
                return torch.full_like(output, token)

            observer = llm.model_runner.model.lm_head.register_forward_hook(observe)
            forcing = llm.model_runner.sampler.register_forward_hook(force_token) if baseline else None
            try:
                outputs = llm.generate([tokenized], sp, use_tqdm=False)
            finally:
                observer.remove()
                if forcing:
                    forcing.remove()
            if len(logits) != args.tokens:
                raise AssertionError("unexpected number of validation steps")
            if baseline is None:
                torch.save({"model": identity_key, "prompt_sha256": prompt_key,
                            "tokens": outputs[0]["token_ids"], "logits": logits}, destination / f"prompt-{index}.pt")
                reports.append({"prompt": index, "status": "recorded", "tokens": outputs[0]["token_ids"]})
                continue
            steps = []
            for step, (actual, expected) in enumerate(zip(logits, baseline["logits"])):
                actual, expected = actual.float(), expected.float()
                delta = (actual - expected).abs()
                logits_ok = bool(torch.isfinite(actual).all() and torch.allclose(actual, expected, atol=2e-2, rtol=2e-2))
                top = expected[0].topk(2)
                steps.append({"step": step, "max_abs_error": delta.max().item(),
                              "mse": delta.square().mean().item(), "logits_close": logits_ok,
                              "baseline_token": baseline["tokens"][step], "candidate_argmax": predicted[step],
                              "baseline_top2_gap": (top.values[0] - top.values[1]).item()})
            divergence = first_difference(predicted, baseline["tokens"])
            report = {"prompt": index, "steps": steps, "first_forced_argmax_difference": divergence,
                      "passed": divergence is None and all(s["logits_close"] for s in steps)}
            if divergence is not None:
                llm.scheduler.block_manager = BlockManager(64, 256)
                free = llm.generate([tokenized], sp, use_tqdm=False)[0]["token_ids"]
                report["free_run_tokens"] = free
                report["first_free_run_difference"] = first_difference(free, baseline["tokens"])
                # Preserve raw tensors at the first divergence for diagnosis.
                torch.save({"step": divergence, "actual": logits[divergence],
                            "expected": baseline["logits"][divergence]}, destination / f"divergence-{index}.pt")
            reports.append(report)
            write_json(destination / f"prompt-{index}.json", report)
        success = all(r.get("passed", True) for r in reports)
        write_json(destination / "summary.json", {"status": "passed" if success else "failed",
                   "backend": args.backend, "mode": args.mode, "model": identity_key, "prompts": reports})
        if not success:
            raise SystemExit(1)
    except Exception as exc:
        write_json(destination / "failure.json", {"error": repr(exc), "completed_prompts": len(reports)})
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
