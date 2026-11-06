"""Explicit preflight before capture; only the partition numerical update varies."""
import re
from fragment_asm_d256 import marker_asm
from fragment_partition_asm_d256 import partition_state_asm
from warp_ablation_common import merge


def geometry(spec, plan, variant):
    if variant not in ('baseline', 'shared'):
        raise ValueError('variant must be baseline or shared')
    expected = {64:256, 128:128, 256:64}.get(spec.head_dim)
    if expected is None or plan.tile_tokens != expected:
        raise ValueError('experiment requires D64/TILE256, D128/TILE128 or D256/TILE64')
    if not 1 <= spec.num_heads // spec.num_kv_heads <= 16:
        raise ValueError('query/KV group must be in [1,16]')


def validate_layout(text, tile, *, probe=False):
    layout = re.search(r'^#mma = (.*)$', text, re.M)
    if not layout or not all(s in layout[1] for s in ('versionMajor = 2', 'warpsPerCTA = [1, 4]')):
        raise ValueError('unsupported MMA layout; candidate must not launch')
    calls = [s for s in text.splitlines() if 'elementwise_inline_asm' in s and 'fragment_p[' in s]
    tensor = 'tensor<16x'+str(tile)+'xf32, #mma>'
    if len(calls) != 1 or calls[0].count(tensor) != 4:
        raise ValueError('state asm must have three MMA inputs and one MMA output')
    if probe:
        markers = [s for s in text.splitlines() if 'elementwise_inline_asm' in s and 'mad.lo.u32 $0, tid,' in s]
        if len(markers) != 1 or tensor not in markers[0]:
            raise ValueError('probe input must keep the audited MMA layout')


def prepare(q, k, v, table, lengths, resources, variant, *, tags=None, fp_fusion=True):
    geometry(resources.spec, resources.plan, variant)
    import torch
    if torch.cuda.get_device_capability(q.device) != (8,0):
        raise ValueError('fragment experiment is audited only on sm80')
    from paged_decode.matched_splitkv import _normalized_partition
    from paged_decode.experimental_fragment_partition import _fragment_partition
    spec, plan = resources.spec, resources.plan
    scale = spec.head_dim ** -.5
    grid = (spec.max_batch, spec.num_kv_heads, plan.splits)
    inputs = (q, k, v, table, lengths)
    args = (*inputs, resources.partials, resources.lse)
    options = dict(Q0=q.stride(0), Q1=q.stride(1), TABLE0=table.stride(0), HQ=spec.num_heads,
                   HKV=spec.num_kv_heads, D=spec.head_dim, PARTS=plan.splits, PART_SIZE=plan.tokens_per_split,
                   SCALE=scale, SCALE_LOG2=scale*1.4426950408889634, TILE=plan.tile_tokens, num_warps=4,
                   enable_fp_fusion=fp_fusion)
    kernel = _normalized_partition
    if variant == 'shared':
        asm, constraints, pack = partition_state_asm(plan.tile_tokens, fp_fusion=fp_fusion)
        tag_asm, tag_constraints, _ = marker_asm(plan.tile_tokens)
        kernel = _fragment_partition
        args += (tags if tags is not None else q,)
        options.update(ASM=asm, CONSTRAINTS=constraints, PACK=pack, PROBE=tags is not None,
                       TAG_ASM=tag_asm, TAG_CONSTRAINTS=tag_constraints)
    elif tags is not None:
        raise ValueError('tags are supported only by the shared candidate')
    compiled = kernel.warmup(*args, grid=grid, **options)
    if variant == 'shared':
        validate_layout(compiled.asm['ttgir'], plan.tile_tokens, probe=tags is not None)
        if compiled.metadata.num_warps != 4:
            raise ValueError('candidate must use exactly 128 threads')
        static_expected = 16*plan.tile_tokens*4+256
        bounded = re.findall(r'\.shared\s+\.align\s+4\s+\.b8\s+(fragment_(?:p|sum))\[(\d+)\]',compiled.asm['ptx'])
        if sorted(bounded) != [('fragment_p',str(16*plan.tile_tokens*4)),('fragment_sum','256')]:
            raise ValueError('unexpected static shared allocation')
        # Conservative A100 per-block opt-in limit, verified in this project.
        if compiled.metadata.shared + static_expected > 166912:
            raise ValueError('candidate exceeds audited sm80 shared budget')
    compiled._init_handles()
    resources._fragment_prepared = (variant, inputs, lambda: kernel[grid](*args, **options), compiled)
    return compiled


def partition(q, k, v, table, lengths, resources, variant):
    saved = getattr(resources, '_fragment_prepared', None)
    if saved is None or saved[0] != variant:
        raise ValueError('prepare this variant before eager execution or Graph capture')
    if any(a is not b for a,b in zip((q,k,v,table,lengths), saved[1])):
        raise ValueError('input objects changed; run prepare again')
    return saved[2]()


def complete(q, k, v, table, lengths, resources, variant):
    partition(q,k,v,table,lengths,resources,variant)
    merge(resources)
    return resources.out


def strict_equal(torch, a, b):
    """Compare representations as well as dtype/shape, including signed zeros."""
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
