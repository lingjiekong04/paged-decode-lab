"""Random and saved-activation tests for explicit/native-geometry split-KV."""
import argparse
from dataclasses import asdict
from pathlib import Path
from _common import fingerprint, require_gpu, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="optional native-activations.pt from long diagnosis")
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    from flash_attn import flash_attn_with_kvcache
    from paged_decode.matched_splitkv import allocate_matched_resources, matched_decode
    from paged_decode.spec import DecodeSpec
    from paged_decode.torch_reference import dense_reference
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "environment.json", fingerprint())

    def compare(a, b):
        delta = a.float() - b.float()
        return {"max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
            "bitwise_equal": torch.equal(a, b), "fraction_unequal": (a != b).float().mean().item(),
            "close_2e-2": bool(torch.isfinite(a).all() and torch.allclose(a.float(), b.float(), atol=.02, rtol=.02))}
    rows = []
    if args.input:
        data = torch.load(args.input, map_location="cpu", weights_only=True)["snapshots"]
        cases = [(name, v["q"], v["k"], v["v"], args.capacity, v["returned_out"])
                 for name, v in data.items() if "/attention" in name]
    else:
        torch.manual_seed(2026)
        cases = []
        for batch, heads, kv, dim, cap, length in ((8,8,2,256,1024,513), (8,16,4,256,1024,587),
                (1,16,4,256,2176,2081), (1,8,2,256,2176,2049), (1,4,1,64,1025,1025), (3,4,2,128,1025,1025)):
            cases.append((f"b{batch}-h{heads}-d{dim}-c{cap}-l{length}",
                torch.randn(batch, heads, dim, dtype=torch.bfloat16, device="cpu"),
                torch.randn(batch, length, kv, dim, dtype=torch.bfloat16, device="cpu"),
                torch.randn(batch, length, kv, dim, dtype=torch.bfloat16, device="cpu"), cap, None))
    for name, cpu_q, keys, vals, cap, saved in cases:
        batch, length, kv, dim = keys.shape
        q = cpu_q.cuda()
        spec = DecodeSpec(batch, q.shape[1], kv, dim, cap)
        table = torch.randperm(batch * spec.table_width, device="cuda").int().view(batch, spec.table_width)
        k, v = (torch.zeros(batch * spec.table_width, 256, kv, dim, dtype=torch.bfloat16, device="cuda") for _ in range(2))
        pos = torch.arange(length, device="cuda")
        page = table[:, pos // 256].long()
        k[page, pos[None, :] % 256], v[page, pos[None, :] % 256] = keys.cuda(), vals.cuda()
        lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
        resources = allocate_matched_resources(spec, "cuda")
        actual = matched_decode(q, k, v, table, lengths, resources=resources).clone()
        native = lambda n: flash_attn_with_kvcache(q.unsqueeze(1), k, v, block_table=table,
            cache_seqlens=lengths, causal=True, num_splits=n).squeeze(1)
        auto, explicit = native(0), native(resources.plan.splits)
        row = {"location": name, "plan": asdict(resources.plan), "matched_vs_auto": compare(actual, auto),
            "predicted_fixed_vs_auto": compare(explicit, auto), "matched_vs_fixed": compare(actual, explicit),
            "matched_vs_fp32": compare(actual, dense_reference(q,k,v,table,lengths))}
        if saved is not None:
            row["reconstructed_auto_vs_saved"] = compare(auto.cpu(), saved)
        rows.append(row)
        print(name, "splits",resources.plan.splits,"matched",row["matched_vs_auto"],flush=True)
    write_json(args.output / "summary.json", {"status":"completed","rows":rows})


if __name__ == "__main__":
    main()
