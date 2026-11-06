"""Controlled baseline/late_v launches, both using four partition and merge warps."""
from warp_ablation_common import merge


def partition(q,k,v,table,lengths,resources,variant):
    if variant not in ('baseline','late_v'):
        raise ValueError('variant must be baseline or late_v')
    from paged_decode.matched_splitkv import _normalized_partition
    from paged_decode.experimental_late_v import _partition_late_v
    kernel=_normalized_partition if variant=='baseline' else _partition_late_v
    spec,plan=resources.spec,resources.plan;scale=spec.head_dim**-.5
    return kernel[(spec.max_batch,spec.num_kv_heads,plan.splits)](
        q,k,v,table,lengths,resources.partials,resources.lse,
        Q0=q.stride(0),Q1=q.stride(1),TABLE0=table.stride(0),HQ=spec.num_heads,HKV=spec.num_kv_heads,
        D=spec.head_dim,PARTS=plan.splits,PART_SIZE=plan.tokens_per_split,SCALE=scale,
        SCALE_LOG2=scale*1.4426950408889634,TILE=plan.tile_tokens,num_warps=4)


def complete(q,k,v,table,lengths,resources,variant):
    partition(q,k,v,table,lengths,resources,variant);merge(resources)
    return resources.out
