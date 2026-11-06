"""One backend/mode per process. Startup, compilation and warmup are untimed."""
import argparse
import hashlib
import time
from pathlib import Path
from _common import require_gpu, fingerprint, write_json, summarize, model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("flash_attn", "triton_base", "triton_splitk"), required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--kv-blocks", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch, args.input_tokens, args.output_tokens, args.kv_blocks, args.repeats) <= 0:
        parser.error("all numeric arguments must be positive")
    maximum = args.input_tokens + args.output_tokens
    if args.kv_blocks < args.batch * ((maximum + 255) // 256):
        parser.error("KV pool too small for all requests; increase --kv-blocks to avoid preemption confounds")
    torch = require_gpu()
    identity = model_identity(args.model)
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.block_manager import BlockManager
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "environment.json", fingerprint())
    write_json(destination / "arguments.json", {**vars(args), "model_identity": identity})
    llm = None
    rows = []
    try:
        torch.manual_seed(123)
        llm = LLM(str(Path(args.model).resolve()), max_num_seqs=max(32, args.batch),
                  max_num_batched_tokens=max(8192, args.batch * args.input_tokens),
                  max_model_len=maximum, kv_cache_blocks=args.kv_blocks,
                  gpu_memory_utilization=.75, tensor_parallel_size=1,
                  decode_backend=args.backend, enforce_eager=args.mode == "eager",
                  record_decode_timings=True)
        generator = torch.Generator().manual_seed(927)
        vocab = llm.model_runner.config.hf_config.vocab_size
        prompts = torch.randint(100, min(vocab, 30000), (args.batch, args.input_tokens),
                                generator=generator, device="cpu").tolist()
        import json
        prompt_hash = hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest()
        sp = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
        # Full-shape warmup includes the sampler's torch.compile and all decode graphs.
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            # Reset prefix state between repeats with no live requests.
            assert llm.is_finished()
            llm.scheduler.block_manager = BlockManager(args.kv_blocks, 256)
            llm.model_runner.decode_timings.clear()
            torch.manual_seed(123)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = llm.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            timings = [{"milliseconds": a.elapsed_time(b), "batch": batch}
                       for a, b, batch in llm.model_runner.decode_timings]
            count = sum(len(o["token_ids"]) for o in outputs)
            if count != args.batch * args.output_tokens:
                raise AssertionError("unexpected output token count")
            row = {"repeat": repeat, "backend": args.backend, "mode": args.mode,
                   "prompt_sha256": prompt_hash, "kv_blocks": llm.model_runner.config.num_kvcache_blocks,
                   "output_tokens": count, "generation_wall_seconds": elapsed,
                   "output_tokens_per_second": count / elapsed,
                   "decode_model_intervals": timings,
                   "decode_model_ms": summarize([t["milliseconds"] for t in timings]),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "token_ids": [o["token_ids"] for o in outputs],
                   "timing_note": "decode intervals include model forward and lm_head, exclude prepare/sample; event instrumentation enabled"}
            rows.append(row)
            write_json(destination / f"repeat-{repeat}.json", row)
        write_json(destination / "summary.json", {"status": "completed", "rows": rows,
                   "throughput": summarize([r["output_tokens_per_second"] for r in rows])})
    except Exception as exc:
        write_json(destination / "failure.json", {"status": "failed", "error": repr(exc), "completed_repeats": len(rows)})
        raise
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
