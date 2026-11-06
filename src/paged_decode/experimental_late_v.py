"""V-load placement experiment derived from the unchanged matched partition.

Only the source position of the V load changes. All arithmetic and launches stay
fixed. This module is not registered as a production backend.
"""
import triton
import triton.language as tl
from .matched_splitkv import _rcp, _fast_log


@triton.jit
def _partition_late_v(Q, K, V, Tables, Lengths, Partial, LSE,
    Q0: tl.constexpr, Q1: tl.constexpr, TABLE0: tl.constexpr, HQ: tl.constexpr,
    HKV: tl.constexpr, D: tl.constexpr, PARTS: tl.constexpr, PART_SIZE: tl.constexpr,
    SCALE: tl.constexpr, SCALE_LOG2: tl.constexpr, TILE: tl.constexpr):
    b, kh, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group: tl.constexpr = HQ // HKV
    r, d, t = tl.arange(0, 16), tl.arange(0, D), tl.arange(0, TILE)
    heads = kh * group + r
    q = tl.load(Q + b * Q0 + heads[:, None] * Q1 + d[None, :], r[:, None] < group, 0)
    length = tl.load(Lengths + b)
    start, end = part * PART_SIZE, tl.minimum((part + 1) * PART_SIZE, length)
    count = tl.maximum(tl.cdiv(end - start, TILE), 0)
    m = tl.full((16,), -float("inf"), tl.float32)
    lanes = tl.arange(0, 4)
    blocks = tl.arange(0, TILE // 8)
    denom_lanes = tl.full((16, 4), 0., tl.float32)
    acc = tl.full((16, D), 0., tl.float32)
    for index in range(count):
        pos = start + (count - 1 - index) * TILE + t
        valid = pos < end
        page = tl.load(Tables + b * TABLE0 + pos // 256, valid, 0)
        slot = (page * 256 + pos % 256) * HKV + kh
        k = tl.load(K + d[:, None] + slot[None, :] * D, valid[None, :], 0)
        scores = tl.where(valid[None, :], tl.dot(q, k), -float("inf"))
        next_m = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp2((m - next_m) * SCALE_LOG2)
        p = tl.exp2(tl.fma(scores, SCALE_LOG2, -next_m[:, None] * SCALE_LOG2))
        denom_lanes = denom_lanes * alpha[:, None]
        even, odd = tl.split(tl.reshape(p, (16, TILE // 8, 4, 2)))
        for block in tl.static_range(TILE // 8):
            denom_lanes = denom_lanes + tl.sum(tl.where(blocks[None, :, None] == block, even, 0.), 1)
            denom_lanes = denom_lanes + tl.sum(tl.where(blocks[None, :, None] == block, odd, 0.), 1)
        v = tl.load(V + slot[:, None] * D + d[None, :], valid[:, None], 0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        m = next_m
    s0 = tl.sum(tl.where(lanes[None, :] == 0, denom_lanes, 0.), 1)
    s1 = tl.sum(tl.where(lanes[None, :] == 1, denom_lanes, 0.), 1)
    s2 = tl.sum(tl.where(lanes[None, :] == 2, denom_lanes, 0.), 1)
    s3 = tl.sum(tl.where(lanes[None, :] == 3, denom_lanes, 0.), 1)
    denom = (s0 + s2) + (s1 + s3)
    safe = tl.where(denom > 0, denom, 1.)
    normalized = acc * _rcp(safe)[:, None]
    lse = tl.where(denom > 0, tl.fma(m, SCALE, _fast_log(safe)), -float("inf"))
    address = (b * HQ + heads) * PARTS + part
    tl.store(LSE + address, lse, r < group)
    tl.store(Partial + address[:, None] * D + d[None, :], normalized, r[:, None] < group)


