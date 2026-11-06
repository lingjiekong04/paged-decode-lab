"""Minimal real-kernel target for Nsight Compute; never part of formal timing."""
import argparse
from _common import require_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triton_base", "triton_splitk"), default="triton_base")
    args = parser.parse_args()
    torch = require_gpu()
    from paged_decode.runtime import allocate_resources, paged_decode
    from paged_decode.spec import DecodeSpec
    q = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(32, 256, 2, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.randperm(32, device="cuda").int().view(1, 32)
    lengths = torch.tensor([8192], device="cuda", dtype=torch.int32)
    resources = allocate_resources(DecodeSpec(1, 8, 2, 256, 8192), "cuda", args.backend)
    for _ in range(3):
        paged_decode(q, k, v, table, lengths, resources=resources, backend=args.backend)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
