"""Experimental launch adapter: only partition warp count varies, merge stays at four."""
import math


def partition(q,k,v,table,lengths,resources,warps):
    if type(warps) is not int or warps not in (4,8):
        raise ValueError('partition warps must be 4 or 8')
    from paged_decode.matched_splitkv import _normalized_partition
    spec,plan=resources.spec,resources.plan
    scale=spec.head_dim**-.5
    return _normalized_partition[(spec.max_batch,spec.num_kv_heads,plan.splits)](
        q,k,v,table,lengths,resources.partials,resources.lse,
        Q0=q.stride(0),Q1=q.stride(1),TABLE0=table.stride(0),
        HQ=spec.num_heads,HKV=spec.num_kv_heads,D=spec.head_dim,
        PARTS=plan.splits,PART_SIZE=plan.tokens_per_split,
        SCALE=scale,SCALE_LOG2=scale*1.4426950408889634,TILE=plan.tile_tokens,num_warps=warps)


def merge(resources):
    from paged_decode.matched_splitkv import _lse_merge
    spec,plan=resources.spec,resources.plan
    padded=1<<(plan.splits-1).bit_length()
    return _lse_merge[(spec.max_batch,spec.num_heads)](resources.partials,resources.lse,resources.out,
        HQ=spec.num_heads,D=spec.head_dim,PARTS=plan.splits,PAD_PARTS=padded,LANES=plan.lse_lanes,num_warps=4)


def complete(q,k,v,table,lengths,resources,warps):
    partition(q,k,v,table,lengths,resources,warps)
    merge(resources)
    return resources.out
