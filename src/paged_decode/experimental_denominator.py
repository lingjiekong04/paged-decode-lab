"""Experimental block selection for ordered denominators; not a production backend.

The partition body derives from this project's matched_splitkv.py. Only selection
of probability blocks changes; tile, partition plan and accumulation order remain.
"""
import triton
import triton.language as tl
from .matched_splitkv import _rcp, _fast_log


@triton.jit
def _block_1(x, I: tl.constexpr):
    tl.static_assert(I == 0)
    return tl.reshape(x, (16, 8))


@triton.jit
def _block_2(x, I: tl.constexpr):
    left, right = tl.split(tl.trans(tl.reshape(x, (16, 1, 2, 8)), 0, 1, 3, 2))
    if I % 2 == 0:
        return _block_1(left, I // 2)
    else:
        return _block_1(right, I // 2)


@triton.jit
def _block_4(x, I: tl.constexpr):
    left, right = tl.split(tl.trans(tl.reshape(x, (16, 2, 2, 8)), 0, 1, 3, 2))
    if I % 2 == 0:
        return _block_2(left, I // 2)
    else:
        return _block_2(right, I // 2)


@triton.jit
def _block_8(x, I: tl.constexpr):
    left, right = tl.split(tl.trans(tl.reshape(x, (16, 4, 2, 8)), 0, 1, 3, 2))
    if I % 2 == 0:
        return _block_4(left, I // 2)
    else:
        return _block_4(right, I // 2)


@triton.jit
def _block_16(x, I: tl.constexpr):
    left, right = tl.split(tl.trans(tl.reshape(x, (16, 8, 2, 8)), 0, 1, 3, 2))
    if I % 2 == 0:
        return _block_8(left, I // 2)
    else:
        return _block_8(right, I // 2)


@triton.jit
def _block_32(x, I: tl.constexpr):
    left, right = tl.split(tl.trans(tl.reshape(x, (16, 16, 2, 8)), 0, 1, 3, 2))
    if I % 2 == 0:
        return _block_16(left, I // 2)
    else:
        return _block_16(right, I // 2)


@triton.jit
def _select_block(x, N: tl.constexpr, I: tl.constexpr):
    tl.static_assert(N >= 1)
    tl.static_assert(N <= 32)
    tl.static_assert((N & (N - 1)) == 0)
    tl.static_assert(I >= 0 and I < N)
    if N == 1:
        return _block_1(x, I)
    elif N == 2:
        return _block_2(x, I)
    elif N == 4:
        return _block_4(x, I)
    elif N == 8:
        return _block_8(x, I)
    elif N == 16:
        return _block_16(x, I)
    elif N == 32:
        return _block_32(x, I)


@triton.jit
def _partition_select(Q, K, V, Tables, Lengths, Partial, LSE,
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
    denom_lanes = tl.full((16, 4), 0., tl.float32)
    acc = tl.full((16, D), 0., tl.float32)
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
        denom_lanes = denom_lanes * alpha[:, None]
        for block in tl.static_range(TILE // 8):
            selected = _select_block(tl.reshape(p, (16, TILE // 8, 8)), TILE // 8, block)
            even, odd = tl.split(tl.reshape(selected, (16, 4, 2)))
            denom_lanes = denom_lanes + even
            denom_lanes = denom_lanes + odd
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




