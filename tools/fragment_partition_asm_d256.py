"""Partition-specific contraction of alpha*state + the first even probability.

The previous micro-harness used an already-rounded initial state. The pinned
partition's ptxas default contracts its first multiply/add; preserve that here.
"""
from fragment_asm_d256 import shared_ordered_asm


def partition_state_asm(tile, *, fp_fusion=True):
    code,_,pack=shared_ordered_asm(tile,states=True)
    if fp_fusion:
        code=code.replace('add.rn.f32 a, a, x;',f'fma.rn.f32 a, a, ${3*pack}, x;',1)
        code=code.replace('add.rn.f32 b, b, y;',f'fma.rn.f32 b, b, ${3*pack+2}, y;',1)
    else:
        code=code.replace(f'mov.f32 b, ${2*pack+2};',f'mov.f32 b, ${2*pack+2};\nmul.rn.f32 a, a, ${3*pack};\nmul.rn.f32 b, b, ${3*pack+2};')
    return code,','.join(['=f']*pack+['f']*(3*pack)+['~{memory}']),pack
