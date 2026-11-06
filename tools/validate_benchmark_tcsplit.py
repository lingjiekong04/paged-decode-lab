"""Validate public TC split-K and optionally benchmark long-input generation.

One model load; short fixed-history regression uses a three-partition capacity
when --benchmark is enabled, then native/TC/TC-split are timed on the same KV pool.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from _common import ROOT, fingerprint, model_identity, require_gpu, summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.input_tokens, args.tokens, args.repeats) < 1:
        parser.error("positive lengths/repeats required")
    torch = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    from paged_decode.hybrid import HybridRunner
    from paged_decode.runtime import allocate_resources
    args.output.mkdir(parents=True, exist_ok=False)
    identity = model_identity(args.model)
    identity = {k: identity[k] for k in ("repo_id", "revision", "weights", "files")}
    write_json(args.output / "environment.json", fingerprint())
    write_json(args.output / "arguments.json", {**vars(args), "reference": str(args.reference), "output": str(args.output),
        "model_identity": identity, "torch_threads_validation": 4, "torch_threads_benchmark": original_threads,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    maximum = max(1024, args.input_tokens + args.tokens) if args.benchmark else 1024
    runner = HybridRunner(args.model, backend="triton_tc_splitk", batch=1, max_context=maximum)
    write_json(args.output / "runtime.json", runner.description())

    def configure(backend, mode):
        runner.graph = None
        for name in ("graph_logits", "graph_tokens", "graph_position"):
            if hasattr(runner, name):
                delattr(runner, name)
        gc.collect()
        torch.cuda.empty_cache()
        runner.backend, runner.mode = backend, mode
        runner.resources = None if backend == "flash_attn" else allocate_resources(runner.spec, runner.device, backend)
        if mode == "graph":
            runner._capture()

    prompts = json.loads((ROOT / "configs/prompts.json").read_text())
    validation = []
    for mode in ("eager", "graph"):
        configure("triton_tc_splitk", mode)
        reports = []
        print("VALIDATE", mode, flush=True)
        for index, text in enumerate(prompts):
            encoded = runner.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False)
            baseline = torch.load(args.reference / f"prompt-{index}.pt", map_location="cpu", weights_only=True)
            if baseline["model"] != identity or baseline["prompt_sha256"] != hashlib.sha256(json.dumps(encoded).encode()).hexdigest():
                raise ValueError("model/prompt identity differs from baseline")
            result = runner.generate([encoded], len(baseline["tokens"]), observe=True,
                                     forced=[[t] for t in baseline["tokens"]])
            steps = []
            for step, (actual, expected) in enumerate(zip(result["logits"], baseline["logits"])):
                delta = actual.float() - expected.float()
                steps.append({"step": step, "max_abs": delta.abs().max().item(),
                    "rmse": delta.square().mean().sqrt().item(), "bitwise_equal": torch.equal(actual, expected),
                    "logits_close": bool(torch.isfinite(actual).all() and torch.allclose(actual.float(), expected.float(), atol=.02, rtol=.02)),
                    "baseline_token": baseline["tokens"][step], "candidate_token": result["predicted_ids"][0][step]})
            report = {"prompt": index, "prompt_tokens": len(encoded), "steps": steps,
                "passed": all(r["logits_close"] and r["baseline_token"] == r["candidate_token"] for r in steps)}
            reports.append(report)
            write_json(args.output / f"validate-{mode}-prompt-{index}.json", report)
        flat = [s for r in reports for s in r["steps"]]
        row = {"mode": mode, "passed": all(r["passed"] for r in reports), "tested_steps": len(flat),
            "bitwise_equal_steps": sum(r["bitwise_equal"] for r in flat),
            "argmax_differences": sum(r["baseline_token"] != r["candidate_token"] for r in flat),
            "logits_failed_steps": sum(not r["logits_close"] for r in flat)}
        validation.append(row)
        print(json.dumps(row), flush=True)
    passed = all(r["passed"] for r in validation)
    write_json(args.output / "validation.json", {"status": "passed" if passed else "numerical_mismatch", "cases": validation})
    if not passed:
        raise SystemExit(2)

    benchmark = []
    if args.benchmark:
        torch.set_num_threads(original_threads)
        generator = torch.Generator(device="cpu").manual_seed(927)
        inputs = torch.randint(100, 30000, (1, args.input_tokens), generator=generator, device="cpu").tolist()
        prompt_hash = hashlib.sha256(json.dumps(inputs).encode()).hexdigest()
        native_tokens = None
        for backend in ("flash_attn", "triton_tc", "triton_tc_splitk"):
            for mode in ("eager", "graph"):
                configure(backend, mode)
                runner.generate(inputs, min(2, args.tokens), observe=True)
                warm = runner.generate(inputs, args.tokens)
                if native_tokens is None:
                    native_tokens = warm["token_ids"][0]
                rows = []
                print("BENCHMARK", backend, mode, flush=True)
                for repeat in range(args.repeats):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = runner.generate(inputs, args.tokens, timing=True)
                    elapsed = time.perf_counter() - started
                    first = next((i for i, (a, b) in enumerate(zip(result["token_ids"][0], native_tokens)) if a != b), None)
                    row = {"backend": backend, "mode": mode, "repeat": repeat,
                        "generation_wall_seconds": elapsed, "output_tokens_per_second": args.tokens / elapsed,
                        "decode_model_ms_raw": result["decode_ms"], "decode_model_ms": summarize(result["decode_ms"]),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "kv_pages_per_full_layer": runner.cache.width, "partition_capacity": runner.spec.partitions,
                        "workspace_bytes": runner.spec.split_workspace_bytes if backend == "triton_tc_splitk" else 0,
                        "prompt_sha256": prompt_hash, "token_ids": result["token_ids"][0],
                        "first_free_greedy_difference_vs_native": first}
                    rows.append(row)
                    write_json(args.output / f"bench-{backend}-{mode}-r{repeat}.json", row)
                summary = {"backend": backend, "mode": mode,
                    "throughput": summarize([r["output_tokens_per_second"] for r in rows]),
                    "greedy_matches_native_all_repeats": all(r["first_free_greedy_difference_vs_native"] is None for r in rows)}
                benchmark.append(summary)
                print(json.dumps(summary), flush=True)
    write_json(args.output / "summary.json", {"status": "completed", "validation": validation,
        "benchmark": benchmark, "max_context": maximum, "partition_capacity": runner.spec.partitions,
        "note": "strict fixed short prompts and synthetic long free generation are separate measurements; same KV budget across backends"})


if __name__ == "__main__":
    main()
