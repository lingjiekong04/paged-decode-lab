"""sm80 resource queries and controlled launches; partition arithmetic is unchanged."""
import ctypes
from dataclasses import asdict
import re

from fragment_d256_common import prepare, strict_equal


class Driver:
    def __init__(self):
        self.lib = ctypes.CDLL('libcuda.so.1')
        self.lib.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        self.lib.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'CUDA driver error {code}')

    def attr(self, function, attribute):
        value = ctypes.c_int()
        self.check(self.lib.cuFuncGetAttribute(ctypes.byref(value), attribute, function))
        return value.value

    def device(self, attribute):
        value = ctypes.c_int()
        self.check(self.lib.cuDeviceGetAttribute(ctypes.byref(value), attribute, 0))
        return value.value

    def occupancy(self, function, dynamic, threads=128):
        value = ctypes.c_int()
        self.check(self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(value), function, threads, dynamic))
        return value.value

    def allow_dynamic(self, function, size):
        if self.attr(function, 8) < size:
            self.check(self.lib.cuFuncSetAttribute(function, 8, size))

    def device_resources(self):
        return {name:self.device(i) for name,i in (
            ('shared_bytes_per_sm',81),('registers_per_sm',82),('max_threads_per_sm',39),
            ('sm_count',16),('shared_optin_per_block',97),('reserved_shared_per_block',111))}


def synthetic(torch, context=2064):
    from paged_decode.spec import DecodeSpec
    torch.manual_seed(20261105)
    spec=DecodeSpec(1,16,4,256,2112)
    q=torch.randn(1,16,256,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(9,256,4,256,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    table=torch.randperm(9,device='cuda').int().view(1,9)
    lengths=torch.tensor([context],device='cuda',dtype=torch.int32)
    return spec,(q,k,v,table,lengths)


def build(spec, tensors, variant, driver):
    """Use the same JIT launch arguments as the frozen D256 preflight adapter."""
    from paged_decode.matched_splitkv import allocate_matched_resources, _normalized_partition
    from paged_decode.experimental_fragment_partition import _fragment_partition
    from fragment_partition_asm_d256 import partition_state_asm
    from fragment_asm_d256 import marker_asm
    q,k,v,table,lengths=tensors
    r=allocate_matched_resources(spec,'cuda',num_splits=36)
    compiled=prepare(*tensors,r,variant)
    args=(*tensors,r.partials,r.lse)
    opts=dict(Q0=q.stride(0),Q1=q.stride(1),TABLE0=table.stride(0),HQ=spec.num_heads,
              HKV=spec.num_kv_heads,D=256,PARTS=36,PART_SIZE=64,SCALE=256**-.5,
              SCALE_LOG2=(256**-.5)*1.4426950408889634,TILE=64,num_warps=4,enable_fp_fusion=True)
    kernel=_normalized_partition
    if variant=='shared':
        code,constraints,pack=partition_state_asm(64,fp_fusion=True)
        tag_code,tag_constraints,_=marker_asm(64)
        args+=(q,)
        opts.update(ASM=code,CONSTRAINTS=constraints,PACK=pack,PROBE=False,
                    TAG_ASM=tag_code,TAG_CONSTRAINTS=tag_constraints)
        kernel=_fragment_partition
    if tuple(compiled.packed_metadata)[:3] != (4,1,compiled.metadata.shared):
        raise ValueError('unrecognized Triton launch metadata')
    packed=tuple(compiled.packed_metadata)

    def launch(parts=36, dynamic=None):
        # Only dynamic shared reservation changes. No PTX, threads, inputs,
        # register limits or numerical instructions are modified.
        size=packed[2] if dynamic is None else dynamic
        if size < packed[2]:
            raise ValueError('cannot reduce actual dynamic shared requirement')
        driver.allow_dynamic(compiled.function,size)
        replacement=(*packed[:2],size,*packed[3:])
        saved=compiled.packed_metadata
        compiled.packed_metadata=replacement
        try:
            result=kernel[(1,4,parts)](*args,**opts)
            if result is not compiled:
                raise RuntimeError('JIT selected a different compiled specialization')
            return result
        finally:
            compiled.packed_metadata=saved
    return r,compiled,launch


def resource_record(compiled,driver):
    metadata=compiled.metadata
    static=driver.attr(compiled.function,1)
    dynamic=metadata.shared
    stack=driver.attr(compiled.function,3)
    return {'registers_per_thread':driver.attr(compiled.function,4),'triton_spill_count':compiled.n_spills,
            'local_bytes_per_thread':stack,'dynamic_shared_bytes':dynamic,'static_shared_bytes':static,
            'reserved_shared_bytes_per_block':driver.device(111),
            'driver_max_active_blocks_per_sm':driver.occupancy(compiled.function,dynamic),
            'threads':128,'metadata':str(metadata),'plan_dynamic_shared_field':dynamic,
            'ptx_barrier_sites':len(re.findall(r'bar\.sync',compiled.asm['ptx']))}
