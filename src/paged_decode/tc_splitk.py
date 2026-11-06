"""Tensor Core paged decode with fixed 1024-token partitions and FP32 merge.

The first stage preserves ordered per-lane denominators. It stores unnormalized
FP32 partials; maxima are stored in scaled score units for the common merge.
"""
import triton
import triton.language as tl
from .kernels import _merge
from .runtime import _validate
from .spec import PARTITION_SIZE


@triton.jit
def _tc_partition(Q, K, V, Tables, Lengths, Maxima, Denoms, Accums,
                  Q0: tl.constexpr, Q1: tl.constexpr, TABLE0: tl.constexpr,
                  HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                  PARTS: tl.constexpr, SCALE: tl.constexpr, SCALE_LOG2: tl.constexpr,
                  TILE: tl.constexpr, PART_SIZE: tl.constexpr):
    b, kh, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group: tl.constexpr = HQ // HKV
    r, d, t = tl.arange(0, 16), tl.arange(0, D), tl.arange(0, TILE)
    heads = kh * group + r
    q = tl.load(Q + b * Q0 + heads[:, None] * Q1 + d[None, :], r[:, None] < group, 0)
    length = tl.load(Lengths + b)
    start = part * PART_SIZE
    end = tl.minimum(start + PART_SIZE, length)
    count = tl.maximum(tl.cdiv(end - start, TILE), 0)
    m = tl.full((16,), -float("inf"), tl.float32)
    lane_denom = tl.full((16, 4), 0., tl.float32)
    acc = tl.full((16, D), 0., tl.float32)
    lanes, blocks = tl.arange(0, 4), tl.arange(0, TILE // 8)
    for index in range(count):
        pos = start + (count - 1 - index) * TILE + t
        valid = pos < end
        page = tl.load(Tables + b * TABLE0 + pos // 256, valid, 0)
        slot = (page * 256 + pos % 256) * HKV + kh
        k = tl.load(K + d[:, None] + slot[None, :] * D, valid[None, :], 0)
        v = tl.load(V + slot[:, None] * D + d[None, :], valid[:, None], 0)
        scores = tl.where(valid[None, :], tl.dot(q, k), -float("inf"))
        next_m = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp2((m - next_m) * SCALE_LOG2)
        p = tl.exp2(tl.fma(scores, SCALE_LOG2, -next_m[:, None] * SCALE_LOG2))
        lane_denom = lane_denom * alpha[:, None]
        even, odd = tl.split(tl.reshape(p, (16, TILE // 8, 4, 2)))
        for block in tl.static_range(TILE // 8):
            pe = tl.sum(tl.where(blocks[None, :, None] == block, even, 0.), 1)
            po = tl.sum(tl.where(blocks[None, :, None] == block, odd, 0.), 1)
            lane_denom = lane_denom + pe
            lane_denom = lane_denom + po
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        m = next_m
    s0 = tl.sum(tl.where(lanes[None, :] == 0, lane_denom, 0.), 1)
    s1 = tl.sum(tl.where(lanes[None, :] == 1, lane_denom, 0.), 1)
    s2 = tl.sum(tl.where(lanes[None, :] == 2, lane_denom, 0.), 1)
    s3 = tl.sum(tl.where(lanes[None, :] == 3, lane_denom, 0.), 1)
    denom = (s0 + s2) + (s1 + s3)
    address = (b * HQ + heads) * PARTS + part
    tl.store(Maxima + address, m * SCALE, r < group)
    tl.store(Denoms + address, denom, r < group)
    tl.store(Accums + address[:, None] * D + d[None, :], acc, r[:, None] < group)


def tc_splitk_decode(q, k, v, tables, lengths, *, resources, scale):
    _validate(q, k, v, tables, lengths, resources, "triton_splitk", scale)
    if scale <= 0:
        raise ValueError("Tensor Core split-K requires a positive softmax scale")
    spec = resources.spec
    b, h, d = q.shape
    if h // spec.num_kv_heads > 16:
        raise ValueError("Tensor Core split-K supports at most 16 query heads per KV head")
    tile = 256 if d == 64 else (128 if d == 128 else 64)
    _tc_partition[(b, spec.num_kv_heads, spec.partitions)](q, k, v, tables, lengths,
        resources.maxima, resources.denoms, resources.accums, Q0=q.stride(0), Q1=q.stride(1),
        TABLE0=tables.stride(0), HQ=h, HKV=spec.num_kv_heads, D=d, PARTS=spec.partitions,
        SCALE=scale, SCALE_LOG2=scale * 1.4426950408889634, TILE=tile,
        PART_SIZE=PARTITION_SIZE, num_warps=4)
    _merge[(b, h)](resources.maxima, resources.denoms, resources.accums, resources.out,
        HQ=h, D=d, PARTS=spec.partitions, PAD_PARTS=triton.next_power_of_2(spec.partitions), num_warps=4)
    return resources.out[:b]
