"""D256 candidate: explicit two-stage pipeline, with the ordered arithmetic frozen."""
import re
from fragment_d256_common import validate_layout, geometry, strict_equal
from fragment_partition_asm_d256 import partition_state_asm
from fragment_asm_d256 import marker_asm
from partition_residency_common import Driver, resource_record
from warp_ablation_common import merge

VARIANTS=('original','shared_s3','shared_s2')


def prepare(q,k,v,table,lengths,resources,variant):
    if variant not in VARIANTS:raise ValueError('unknown explicit resident variant')
    spec,plan=resources.spec,resources.plan
    if spec.head_dim!=256:raise ValueError('resident experiment supports only D256')
    geometry(spec,plan,'shared' if variant!='original' else 'baseline')
    import torch
    if torch.cuda.get_device_capability(q.device)!=(8,0):raise ValueError('sm80 required')
    from paged_decode.matched_splitkv import _normalized_partition
    from paged_decode.experimental_fragment_partition import _fragment_partition
    grid=(spec.max_batch,spec.num_kv_heads,plan.splits)
    args=(q,k,v,table,lengths,resources.partials,resources.lse)
    stage=2 if variant=='shared_s2' else 3
    scale=256**-.5
    options=dict(Q0=q.stride(0),Q1=q.stride(1),TABLE0=table.stride(0),HQ=spec.num_heads,HKV=spec.num_kv_heads,
                 D=256,PARTS=plan.splits,PART_SIZE=plan.tokens_per_split,SCALE=scale,
                 SCALE_LOG2=scale*1.4426950408889634,TILE=64,num_warps=4,num_stages=stage,enable_fp_fusion=True)
    kernel=_normalized_partition
    if variant!='original':
        code,constraints,pack=partition_state_asm(64,fp_fusion=True)
        tag_code,tag_constraints,_=marker_asm(64)
        kernel=_fragment_partition
        args+=(q,)
        options.update(ASM=code,CONSTRAINTS=constraints,PACK=pack,PROBE=False,
                       TAG_ASM=tag_code,TAG_CONSTRAINTS=tag_constraints)
    compiled=kernel.warmup(*args,grid=grid,**options)
    if compiled.metadata.num_warps!=4 or compiled.metadata.num_stages!=stage:
        raise ValueError('unexpected pipeline geometry')
    if variant!='original':
        validate_layout(compiled.asm['ttgir'],64)
        declarations=re.findall(r'\.shared\s+\.align\s+4\s+\.b8\s+(fragment_(?:p|sum))\[(\d+)\]',compiled.asm['ptx'])
        if sorted(declarations)!=[('fragment_p','4096'),('fragment_sum','256')]:
            raise ValueError('ordered-state shared layout changed')
    compiled._init_handles()
    driver=Driver()
    if variant=='shared_s2' and driver.occupancy(compiled.function,compiled.metadata.shared)<2:
        raise ValueError('candidate did not restore two-CTA residency; no fallback')
    saved=(q,k,v,table,lengths)
    def reduced(parts):
        result=kernel[(spec.max_batch,spec.num_kv_heads,parts)](*args,**options)
        if result is not compiled:raise RuntimeError('grid control selected a different compiled kernel')
        return result
    resources._resident_prepared=(variant,saved,lambda:kernel[grid](*args,**options),compiled,reduced)
    return compiled


def partition(q,k,v,table,lengths,resources,variant):
    state=getattr(resources,'_resident_prepared',None)
    if state is None or state[0]!=variant or any(a is not b for a,b in zip((q,k,v,table,lengths),state[1])):
        raise ValueError('prepare the exact variant and input objects before capture')
    return state[2]()


def complete(q,k,v,table,lengths,resources,variant):
    partition(q,k,v,table,lengths,resources,variant)
    merge(resources)
    return resources.out


def launch_grid(q,k,v,table,lengths,resources,variant,parts):
    state=getattr(resources,'_resident_prepared',None)
    if state is None or state[0]!=variant or any(a is not b for a,b in zip((q,k,v,table,lengths),state[1])):
        raise ValueError('prepare exact inputs before grid control')
    if type(parts) is not int or not 1<=parts<=resources.plan.splits:raise ValueError('invalid reduced grid')
    return state[4](parts)
