"""Experimental capacity-aligned split-KV with normalized FP32 partials + LSE.

No native attention calls or fallback. The source of the numerical protocol is
FlashAttention 2.7.4.post1; this Triton implementation is written for this lab.
"""
from dataclasses import dataclass
import math
import torch
import triton
import triton.language as tl
from .runtime import _validate
from .splitkv_plan import make_splitkv_plan


@dataclass
class MatchedResources:
    spec: object
    plan: object
    out: torch.Tensor
    partials: torch.Tensor
    lse: torch.Tensor


def allocate_matched_resources(spec, device, *, num_splits=0):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("matched resources require CUDA")
    plan = make_splitkv_plan(spec, torch.cuda.get_device_properties(device).multi_processor_count,
                            num_splits=num_splits)
    shape = (spec.max_batch, spec.num_heads, plan.splits)
    return MatchedResources(spec, plan,
        torch.empty((spec.max_batch, spec.num_heads, spec.head_dim), device=device, dtype=torch.bfloat16),
        torch.empty((*shape, spec.head_dim), device=device, dtype=torch.float32),
        torch.empty(shape, device=device, dtype=torch.float32))


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[x],
                                    dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _fast_log(x):
    # CUDA __logf uses approximate log2 followed by a rounded ln(2) multiply.
    # Triton 3.2 tl.log lowers to the more accurate libdevice __nv_logf here.
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; lg2.approx.ftz.f32 t, $1; mul.f32 $0, t, 0f3F317218; }",
        constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _normalized_partition(Q, K, V, Tables, Lengths, Partial, LSE,
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
        v = tl.load(V + slot[:, None] * D + d[None, :], valid[:, None], 0)
        scores = tl.where(valid[None, :], tl.dot(q, k), -float("inf"))
        next_m = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp2((m - next_m) * SCALE_LOG2)
        p = tl.exp2(tl.fma(scores, SCALE_LOG2, -next_m[:, None] * SCALE_LOG2))
        denom_lanes = denom_lanes * alpha[:, None]
        even, odd = tl.split(tl.reshape(p, (16, TILE // 8, 4, 2)))
        for block in tl.static_range(TILE // 8):
            denom_lanes = denom_lanes + tl.sum(tl.where(blocks[None, :, None] == block, even, 0.), 1)
            denom_lanes = denom_lanes + tl.sum(tl.where(blocks[None, :, None] == block, odd, 0.), 1)
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


@triton.jit
def _lse_merge(Partial, LSE, Out, HQ: tl.constexpr, D: tl.constexpr,
               PARTS: tl.constexpr, PAD_PARTS: tl.constexpr, LANES: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    lane, d = tl.arange(0, LANES), tl.arange(0, D)
    base = (b * HQ + h) * PARTS
    local_max = tl.full((LANES,), -float("inf"), tl.float32)
    for j in tl.static_range(PAD_PARTS // LANES):
        i = j * LANES + lane
        val = tl.load(LSE + base + i, i < PARTS, -float("inf"))
        local_max = tl.maximum(local_max, val)
    maximum = tl.max(local_max, 0)
    maximum = tl.where(maximum == -float("inf"), 0., maximum)
    local_sum = tl.full((LANES,), 0., tl.float32)
    for j in tl.static_range(PAD_PARTS // LANES):
        i = j * LANES + lane
        val = tl.load(LSE + base + i, i < PARTS, -float("inf"))
        local_sum = local_sum + tl.exp(val - maximum)
    total = tl.sum(local_sum, 0)
    logsum = tl.where(total > 0, _fast_log(tl.where(total > 0, total, 1.)) + maximum, float("inf"))
    acc = tl.full((D,), 0., tl.float32)
    for part in range(PARTS):
        lse = tl.load(LSE + base + part)
        value = tl.load(Partial + (base + part) * D + d)
        weight = tl.exp(lse - logsum)
        acc = tl.fma(weight, value, acc)
    tl.store(Out + (b * HQ + h) * D + d, acc)


def matched_decode(q, k, v, tables, lengths, *, resources, scale=None):
    scale = q.shape[-1] ** -.5 if scale is None else float(scale)
    _validate(q, k, v, tables, lengths, resources, "triton_base", scale)
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("positive finite scale required")
    spec, plan = resources.spec, resources.plan
    if q.shape[0] != spec.max_batch or tables.shape[1] != spec.table_width:
        raise ValueError("matched plan requires its fixed batch and table capacity")
    shape = (spec.max_batch, spec.num_heads, plan.splits)
    for tensor, geometry in ((resources.lse, shape), (resources.partials, (*shape, spec.head_dim))):
        if tensor.shape != geometry or tensor.device != q.device or tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError("invalid matched workspace")
    _normalized_partition[(spec.max_batch, spec.num_kv_heads, plan.splits)](
        q, k, v, tables, lengths, resources.partials, resources.lse,
        Q0=q.stride(0), Q1=q.stride(1), TABLE0=tables.stride(0), HQ=spec.num_heads,
        HKV=spec.num_kv_heads, D=spec.head_dim, PARTS=plan.splits, PART_SIZE=plan.tokens_per_split,
        SCALE=scale, SCALE_LOG2=scale * 1.4426950408889634, TILE=plan.tile_tokens, num_warps=4)
    _lse_merge[(spec.max_batch, spec.num_heads)](resources.partials, resources.lse, resources.out,
        HQ=spec.num_heads, D=spec.head_dim, PARTS=plan.splits,
        PAD_PARTS=triton.next_power_of_2(plan.splits), LANES=plan.lse_lanes, num_warps=4)
    return resources.out
