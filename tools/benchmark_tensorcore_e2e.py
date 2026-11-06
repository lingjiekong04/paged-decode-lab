"""Fresh native/TC eager/Graph comparison; one loaded fixed-batch model.

Timing includes prefill, decode, greedy sampling and output conversion, matching
the existing end-to-end benchmark's definition. Token equality is recorded as a
separate observation, not a task-accuracy score or timing acceptance criterion.
"""
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time
from _common import fingerprint, model_identity, require_gpu, summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch, args.input_tokens, args.tokens, args.repeats) < 1:
        parser.error("positive sizes required")
    torch = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    from paged_decode.hybrid import HybridRunner
    from paged_decode.runtime import allocate_resources
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    identity = model_identity(args.model)
    write_json(destination / "environment.json", fingerprint())
    write_json(destination / "arguments.json", {**vars(args), "model_identity": identity,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    runner = HybridRunner(args.model, backend="flash_attn", mode="eager", batch=args.batch,
                          max_context=max(1024, args.input_tokens + args.tokens))
    generator = torch.Generator(device="cpu").manual_seed(927)
    prompts = torch.randint(100, 30000, (args.batch, args.input_tokens), generator=generator, device="cpu").tolist()
    prompt_hash = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    reports, native_tokens = [], None
    for backend, mode in itertools.product(("flash_attn", "triton_tc"), ("eager", "graph")):
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
        runner.generate(prompts, min(2, args.tokens), observe=True)
        warm = runner.generate(prompts, args.tokens)
        if native_tokens is None:
            native_tokens = warm["token_ids"]
        rows = []
        print("CASE", backend, mode, flush=True)
        for repeat in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = runner.generate(prompts, args.tokens, timing=True)
            elapsed = time.perf_counter() - started
            row = {"backend": backend, "mode": mode, "repeat": repeat,
                "prompt_sha256": prompt_hash, "output_tokens": args.batch * args.tokens,
                "generation_wall_seconds": elapsed, "output_tokens_per_second": args.batch * args.tokens / elapsed,
                "decode_model_ms_raw": result["decode_ms"], "decode_model_ms": summarize(result["decode_ms"]),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "kv_pages_per_full_layer": args.batch * runner.cache.width,
                "token_ids": result["token_ids"],
                "greedy_matches_native": result["token_ids"] == native_tokens,
                "note": "synthetic fixed-length inputs; token equality is separate from timing; no task-accuracy claim"}
            rows.append(row)
            write_json(destination / f"{backend}-{mode}-r{repeat}.json", row)
        report = {"backend": backend, "mode": mode,
            "throughput": summarize([r["output_tokens_per_second"] for r in rows]),
            "greedy_matches_native_all_repeats": all(r["greedy_matches_native"] for r in rows), "rows": rows}
        reports.append(report)
        print(backend, mode, report["throughput"]["median"], "tokens/s", flush=True)
    write_json(destination / "summary.json", {"status": "completed", "model": identity, "cases": reports,
        "prompt_sha256": prompt_hash, "note": "four fresh configurations, same model instance and KV capacity; five repeats each"})


if __name__ == "__main__":
    main()
