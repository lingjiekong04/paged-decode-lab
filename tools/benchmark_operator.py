"""CUDA-event launch intervals, synchronized full calls, optional device-kernel trace.

CUDA event intervals include any GPU idle gaps between eager launches. They are
not pure kernel busy time. --profile emits a separate CUDA kernel-duration sample.
"""
import argparse
import itertools
import time
from pathlib import Path
from _common import require_gpu, fingerprint, write_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new directory for raw results")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="one short shape instead of the 18-shape matrix")
    parser.add_argument("--profile", action="store_true", help="separate untimed profiler run with CUDA kernel events")
    parser.add_argument("--qwen35", action="store_true", help="Qwen3.5 small-model geometry: Hq=8,Hkv=2,D=256")
    parser.add_argument("--qwen35-large", action="store_true", help="Qwen3.5-9B geometry: Hq=16,Hkv=4,D=256")
    parser.add_argument("--precision-variants", nargs="+", default=[], help="explicit experimental BF16-exp candidates")
    parser.add_argument("--include-tensorcore", action="store_true", help="add the public triton_tc backend")
    parser.add_argument("--include-tensorcore-splitk", action="store_true", help="add the public triton_tc_splitk backend")
    parser.add_argument("--quick", action="store_true", help="three shapes: (B,L)=(1,512),(1,8192),(8,2048)")
    args = parser.parse_args()
    if args.repeats <= 0 or args.iterations <= 0:
        parser.error("repeats and iterations must be positive")
    torch = require_gpu()
    from flash_attn import flash_attn_with_kvcache
    from paged_decode.spec import DecodeSpec
    from paged_decode.runtime import allocate_resources, paged_decode
    from paged_decode.torch_reference import dense_reference
    from paged_decode.experimental_precision import VARIANTS, precision_decode
    if any(v not in VARIANTS for v in args.precision_variants):
        parser.error("unknown experimental precision candidate")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "environment.json", fingerprint())
    rows = []
    if args.qwen35 and args.qwen35_large:
        parser.error("choose one model geometry")
    dims = (256,) if args.qwen35 or args.qwen35_large else (64, 128)
    hq, hkv = (16, 4) if args.qwen35_large else (8, 2) if args.qwen35 else (32, 8)
    shapes = [(1, 512, dims[0])] if args.smoke else list(itertools.product((1, 8, 32), (512, 2048, 8192), dims))
    if args.quick:
        if args.smoke:
            parser.error("choose quick or smoke")
        shapes = [(b, n, d) for b, n in ((1, 512), (1, 8192), (8, 2048)) for d in dims]
    try:
        for batch, length, dim in shapes:
            torch.manual_seed(123)
            width = (length + 255) // 256
            pages = batch * width
            q = torch.randn(batch, hq, dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(pages, 256, hkv, dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            table = torch.randperm(pages, device="cuda").to(torch.int32).view(batch, width)
            lens = torch.full((batch,), length, device="cuda", dtype=torch.int32)
            expected = dense_reference(q, k, v, table, lens)
            backends = ("flash_attn", "triton_base", "triton_splitk") + (("triton_tc",) if args.include_tensorcore else ())
            backends += ("triton_tc_splitk",) if args.include_tensorcore_splitk else ()
            for backend in (*backends, *args.precision_variants):
                spec = DecodeSpec(batch, hq, hkv, dim, length)
                allocation_backend = "triton_base" if backend in VARIANTS else backend
                resources = None if backend == "flash_attn" else allocate_resources(spec, "cuda", allocation_backend)

                def call(full=False):
                    if backend == "flash_attn":
                        return flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=lens,
                                                       block_table=table, causal=True).squeeze(1)
                    r = allocate_resources(spec, "cuda", allocation_backend) if full else resources
                    if backend in VARIANTS:
                        return precision_decode(q, k, v, table, lens, resources=r, variant=backend)
                    return paged_decode(q, k, v, table, lens, resources=r, backend=backend)

                for _ in range(10):
                    actual = call()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
                error = (actual.float() - expected).abs()
                errors = {"max_abs": error.max().item(), "mse": error.square().mean().item()}
                del actual, error
                warm_stream = torch.cuda.Stream()
                warm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm_stream):
                    for _ in range(3):
                        call()
                torch.cuda.current_stream().wait_stream(warm_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured_output = call()
                graph.replay()
                torch.testing.assert_close(captured_output.float(), expected, atol=2e-2, rtol=2e-2)
                for repeat in range(args.repeats):
                    torch.cuda.reset_peak_memory_stats()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        call()
                    end.record()
                    end.synchronize()
                    event_us = start.elapsed_time(end) * 1000 / args.iterations
                    start.record()
                    for _ in range(args.iterations):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    graph_us = start.elapsed_time(end) * 1000 / args.iterations
                    full_samples = []
                    for _ in range(args.iterations):
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        call(full=True)
                        torch.cuda.synchronize()
                        full_samples.append((time.perf_counter() - t) * 1e6)
                    row = {"batch": batch, "context": length, "head_dim": dim,
                           "query_heads": hq, "kv_heads": hkv, "backend": backend, "repeat": repeat,
                           "preallocated_event_us": event_us, "full_call_wall_us": full_samples,
                           "graph_event_us": graph_us,
                           "full_call_summary_us": summarize(full_samples),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "errors": errors,
                           "workspace_bytes": spec.split_workspace_bytes if backend in ("triton_splitk", "triton_tc_splitk") else 0,
                           "flash_attn_note": "native API manages its output/workspace in both modes"}
                    rows.append(row)
                    write_json(destination / f"b{batch}-l{length}-d{dim}-{backend}-r{repeat}.json", row)
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(20):
                            call()
                        torch.cuda.synchronize()
                    events = [{"name": e.name, "duration_us": e.device_time_total}
                              for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    write_json(destination / f"profile-b{batch}-l{length}-d{dim}-{backend}.json",
                               {"iterations": 20, "cuda_events": events,
                                "note": "separate profiler run; not mixed into formal latency samples"})
                del resources, graph, captured_output
            del q, k, v, table, lens, expected
        write_json(destination / "summary.json", {"status": "completed", "rows": rows})
    except Exception as exc:
        write_json(destination / "failure.json", {"status": "failed", "error": repr(exc), "completed_rows": len(rows)})
        raise


if __name__ == "__main__":
    main()
