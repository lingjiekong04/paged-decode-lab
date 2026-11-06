"""Single-query paged attention, written for this project.

Each CTA handles a (sequence, query-head[, KV partition]). K/V use NHD pages.
No Tensor Core or Hopper-only instructions are required. See docs/ALGORITHM.md.
"""
import triton
import triton.language as tl


@triton.jit
def _base(Q, K, V, Tables, Lengths, Out,
          Q0: tl.constexpr, Q1: tl.constexpr, TABLE0: tl.constexpr,
          HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
          SCALE: tl.constexpr, TILE: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    t = tl.arange(0, TILE)
    length = tl.load(Lengths + b)
    q = tl.load(Q + b * Q0 + h * Q1 + d).to(tl.float32)
    m = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0, tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    for tile in range(tl.cdiv(length, TILE)):
        pos = tile * TILE + t
        valid = pos < length
        page = tl.load(Tables + b * TABLE0 + pos // 256, valid, 0)
        offsets = ((page[:, None] * 256 + pos[:, None] % 256) * HKV
                   + h // (HQ // HKV)) * D + d[None, :]
        keys = tl.load(K + offsets, valid[:, None], 0).to(tl.float32)
        vals = tl.load(V + offsets, valid[:, None], 0).to(tl.float32)
        scores = tl.sum(keys * q[None, :], 1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        next_m = tl.maximum(m, tl.max(scores, 0))
        alpha = tl.exp(m - next_m)
        p = tl.exp(scores - next_m)
        acc = acc * alpha + tl.sum(p[:, None] * vals, 0)
        denom = denom * alpha + tl.sum(p, 0)
        m = next_m
    # Graph padding rows have length zero; never divide by zero.
    result = acc / tl.where(denom > 0, denom, 1.)
    tl.store(Out + (b * HQ + h) * D + d, result)


@triton.jit
def _split(Q, K, V, Tables, Lengths, Maxima, Denoms, Accums,
           Q0: tl.constexpr, Q1: tl.constexpr, TABLE0: tl.constexpr,
           HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
           PARTS: tl.constexpr, SCALE: tl.constexpr,
           TILE: tl.constexpr, PART_SIZE: tl.constexpr):
    b, h, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, D)
    t = tl.arange(0, TILE)
    length = tl.load(Lengths + b)
    q = tl.load(Q + b * Q0 + h * Q1 + d).to(tl.float32)
    m = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0, tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    start = part * PART_SIZE
    if start < length:
        end = tl.minimum(start + PART_SIZE, length)
        for tile in range(tl.cdiv(end - start, TILE)):
            pos = start + tile * TILE + t
            valid = pos < end
            page = tl.load(Tables + b * TABLE0 + pos // 256, valid, 0)
            offsets = ((page[:, None] * 256 + pos[:, None] % 256) * HKV
                       + h // (HQ // HKV)) * D + d[None, :]
            keys = tl.load(K + offsets, valid[:, None], 0).to(tl.float32)
            vals = tl.load(V + offsets, valid[:, None], 0).to(tl.float32)
            scores = tl.sum(keys * q[None, :], 1) * SCALE
            scores = tl.where(valid, scores, -float("inf"))
            next_m = tl.maximum(m, tl.max(scores, 0))
            alpha = tl.exp(m - next_m)
            p = tl.exp(scores - next_m)
            acc = acc * alpha + tl.sum(p[:, None] * vals, 0)
            denom = denom * alpha + tl.sum(p, 0)
            m = next_m
    index = (b * HQ + h) * PARTS + part
    tl.store(Maxima + index, m)
    tl.store(Denoms + index, denom)
    tl.store(Accums + index * D + d, acc)


@triton.jit
def _merge(Maxima, Denoms, Accums, Out,
           HQ: tl.constexpr, D: tl.constexpr, PARTS: tl.constexpr,
           PAD_PARTS: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, PAD_PARTS)
    d = tl.arange(0, D)
    index = (b * HQ + h) * PARTS + s
    l = tl.load(Denoms + index, s < PARTS, 0)
    m = tl.load(Maxima + index, s < PARTS, -float("inf"))
    a = tl.load(Accums + index[:, None] * D + d[None, :], (s < PARTS)[:, None], 0)
    # Replace the all-empty maximum before subtraction, avoiding -inf - -inf.
    maximum = tl.max(m, 0)
    maximum = tl.where(maximum == -float("inf"), 0., maximum)
    factors = tl.exp(m - maximum)
    total = tl.sum(l * factors, 0)
    result = tl.sum(a * factors[:, None], 0) / tl.where(total > 0, total, 1.)
    tl.store(Out + (b * HQ + h) * D + d, result)
