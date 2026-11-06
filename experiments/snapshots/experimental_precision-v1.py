"""Explicit experimental BF16-exp attention paths, outside production BACKENDS.

Scalar and Tensor Core variants isolate probability precision from dot-product
execution. No calls to FlashAttention, no fallback, no model-specific patches.
Based on online-softmax equations; see THIRD_PARTY.md for upstream references.
"""
import triton
import triton.language as tl

from .runtime import _validate


VARIANTS = ("scalar_bf16", "tensorcore_bf16")


@triton.jit
def _scalar(Q, K, V, Tables, Lengths, Out,
            Q0: tl.constexpr, Q1: tl.constexpr, TABLE0: tl.constexpr,
            HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
            SCALE_LOG2: tl.constexpr, TILE: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    d, t = tl.arange(0, D), tl.arange(0, TILE)
    length = tl.load(Lengths + b)
    q = tl.load(Q + b * Q0 + h * Q1 + d).to(tl.float32)
    m = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0., tl.float32)
    acc = tl.full((D,), 0., tl.float32)
    count = tl.cdiv(length, TILE)
    for index in range(count):
        pos = (count - 1 - index) * TILE + t
        valid = pos < length
        page = tl.load(Tables + b * TABLE0 + pos // 256, valid, 0)
        offset = ((page[:, None] * 256 + pos[:, None] % 256) * HKV + h // (HQ // HKV)) * D + d[None, :]
        k = tl.load(K + offset, valid[:, None], 0).to(tl.float32)
        v = tl.load(V + offset, valid[:, None], 0).to(tl.float32)
        scores = tl.where(valid, tl.sum(k * q[None, :], 1), -float("inf"))
        next_m = tl.maximum(m, tl.max(scores, 0))
        alpha = tl.exp2((m - next_m) * SCALE_LOG2)
        p = tl.exp2(tl.fma(scores, SCALE_LOG2, -next_m * SCALE_LOG2))
        denom = denom * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p.to(tl.bfloat16).to(tl.float32)[:, None] * v, 0)
        m = next_m
    tl.store(Out + (b * HQ + h) * D + d, acc / tl.where(denom > 0., denom, 1.))


@triton.jit
def _tensorcore(Q, K, V, Tables, Lengths, Out,
                Q0: tl.constexpr, Q1: tl.constexpr, TABLE0: tl.constexpr,
                HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                SCALE_LOG2: tl.constexpr, TILE: tl.constexpr):
    b, kh = tl.program_id(0), tl.program_id(1)
    group: tl.constexpr = HQ // HKV
    r, d, t = tl.arange(0, 16), tl.arange(0, D), tl.arange(0, TILE)
    heads = kh * group + r
    q = tl.load(Q + b * Q0 + heads[:, None] * Q1 + d[None, :], r[:, None] < group, 0)
    length = tl.load(Lengths + b)
    m = tl.full((16,), -float("inf"), tl.float32)
    denom = tl.full((16,), 0., tl.float32)
    acc = tl.full((16, D), 0., tl.float32)
    count = tl.cdiv(length, TILE)
    for index in range(count):
        pos = (count - 1 - index) * TILE + t
        valid = pos < length
        page = tl.load(Tables + b * TABLE0 + pos // 256, valid, 0)
        slot = (page * 256 + pos % 256) * HKV + kh
        k = tl.load(K + d[:, None] + slot[None, :] * D, valid[None, :], 0)
        v = tl.load(V + slot[:, None] * D + d[None, :], valid[:, None], 0)
        scores = tl.dot(q, k)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_m = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp2((m - next_m) * SCALE_LOG2)
        p = tl.exp2(tl.fma(scores, SCALE_LOG2, -next_m[:, None] * SCALE_LOG2))
        denom = denom * alpha + tl.sum(p, 1)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        m = next_m
    result = acc / tl.where(denom > 0., denom, 1.)[:, None]
    tl.store(Out + (b * HQ + heads[:, None]) * D + d[None, :], result, r[:, None] < group)


def precision_decode(q, k, v, tables, lengths, *, resources, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown experimental precision variant {variant!r}")
    _validate(q, k, v, tables, lengths, resources, "triton_base", q.shape[-1] ** -.5)
    b, h, d = q.shape
    hkv = resources.spec.num_kv_heads
    if variant == "tensorcore_bf16" and h // hkv > 16:
        raise ValueError("experimental Tensor Core path supports at most 16 query heads per KV head")
    tile = 256 if d == 64 else (128 if d == 128 else 64)
    common = dict(Q0=q.stride(0), Q1=q.stride(1), TABLE0=tables.stride(0), HQ=h, HKV=hkv,
                  D=d, SCALE_LOG2=d ** -.5 * 1.4426950408889634, TILE=tile)
    if variant == "scalar_bf16":
        _scalar[(b, h)](q, k, v, tables, lengths, resources.out, **common, num_warps=4)
    else:
        _tensorcore[(b, hkv)](q, k, v, tables, lengths, resources.out, **common, num_warps=4)
    return resources.out[:b]
