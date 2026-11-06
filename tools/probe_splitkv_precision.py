"""Same-input numerical ablation of native auto/fixed split-KV and global TC."""
import argparse
from pathlib import Path
from _common import fingerprint, require_gpu, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    from flash_attn import flash_attn_with_kvcache
    from paged_decode.runtime import allocate_resources, paged_decode
    from paged_decode.spec import DecodeSpec
    from paged_decode.torch_reference import dense_reference
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "environment.json", fingerprint())

    def compare(a, b):
        delta = (a.float() - b.float()).abs()
        return {"max_abs": delta.max().item(), "rmse": delta.square().mean().sqrt().item(),
            "fraction_unequal": (a != b).float().mean().item(), "bitwise_equal": torch.equal(a, b),
            "close_2e-2": bool(torch.isfinite(a).all() and torch.allclose(a.float(), b.float(), atol=.02, rtol=.02))}

    rows = []
    for heads, kv_heads in ((8, 2), (16, 4)):
        for length in (513, 1025, 2049):
            torch.manual_seed(2026)
            batch, dim = 8, 256
            spec = DecodeSpec(batch, heads, kv_heads, dim, length)
            pages = batch * spec.table_width
            q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(pages, 256, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            table = torch.randperm(pages, device="cuda").int().view(batch, spec.table_width)
            lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
            resources = allocate_resources(spec, "cuda", "triton_tc")
            tc = paged_decode(q, k, v, table, lengths, resources=resources, backend="triton_tc").clone()
            native = {n: flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=lengths,
                      block_table=table, causal=True, num_splits=n).squeeze(1) for n in (0, 1, 2, 4, 8)}
            reference = dense_reference(q, k, v, table, lengths)
            row = {"heads": heads, "kv_heads": kv_heads, "batch": batch, "context": length,
                "tc_vs_native": {str(n): compare(tc, tensor) for n, tensor in native.items()},
                "native_auto_vs_fixed": {str(n): compare(native[0], native[n]) for n in (1, 2, 4, 8)},
                "tc_vs_fp32": compare(tc, reference)}
            rows.append(row)
    write_json(output / "summary.json", {"status": "completed", "rows": rows,
        "note": "random same-input operator ablation; not a proof of the cause of any specific model token divergence"})


if __name__ == "__main__":
    main()
