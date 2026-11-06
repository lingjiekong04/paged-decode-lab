"""Minimal stdlib-only Nsight target; load the audited cubin, no model/JIT imports."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('baseline','baseline_padded','shared'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parts',type=int,default=36)
    p.add_argument('--cubin-root',type=Path)
    args=p.parse_args()
    if not 1<=args.parts<=36:p.error('grid must contain 1..36 parts')
    root=Path(__file__).resolve().parents[1]
    lib=C.CDLL('libcuda.so.1')
    ptr=C.c_void_p;u64=C.c_uint64;I=C.c_int;U=C.c_uint
    def api(name,signature,*values):
        fn=getattr(lib,name);fn.argtypes=signature;fn.restype=I
        code=fn(*values)
        if code:raise RuntimeError(f'{name}: CUDA error {code}')
    api('cuInit',[U],0)
    ctx=ptr()
    api('cuCtxCreate_v2',[C.POINTER(ptr),U,I],C.byref(ctx),0,0)
    try:
        variant='shared' if args.variant=='shared' else 'baseline'
        path=(args.cubin_root or root/'results/residency-resources-v1')/f'{variant}.cubin'
        module=ptr();function=ptr()
        api('cuModuleLoad',[C.POINTER(ptr),C.c_char_p],C.byref(module),str(path).encode())
        name=b'_fragment_partition' if variant=='shared' else b'_normalized_partition'
        api('cuModuleGetFunction',[C.POINTER(ptr),ptr,C.c_char_p],C.byref(function),module,name)
        dynamic=79232+(4352 if args.variant=='baseline_padded' else 0)
        api('cuFuncSetCacheConfig',[ptr,I],function,1)
        api('cuFuncSetAttribute',[ptr,I,I],function,8,dynamic)
        allocated=[]
        for size in (8192,4718592,4718592,36,4,589824,2304):
            address=u64()
            api('cuMemAlloc_v2',[C.POINTER(u64),C.c_size_t],C.byref(address),size)
            api('cuMemsetD8_v2',[u64,C.c_ubyte,C.c_size_t],address,0,size)
            allocated.append(address)
        table=(I*9)(2,8,6,0,7,4,3,5,1);length=I(2064)
        api('cuMemcpyHtoD_v2',[u64,ptr,C.c_size_t],allocated[3],C.cast(table,ptr),36)
        api('cuMemcpyHtoD_v2',[u64,ptr,C.c_size_t],allocated[4],C.byref(length),4)
        params=allocated+([allocated[0]] if variant=='shared' else [])
        pointers=(ptr*len(params))(*(C.cast(C.byref(x),ptr) for x in params))
        def launch():
            api('cuLaunchKernel',[ptr,U,U,U,U,U,U,U,ptr,C.POINTER(ptr),C.POINTER(ptr)],
                function,1,4,args.parts,128,1,1,dynamic,None,pointers,None)
        for _ in range(3):launch()
        api('cuCtxSynchronize',[])
        api('cuProfilerStart',[])
        launch()
        api('cuCtxSynchronize',[])
        api('cuProfilerStop',[])
        active=I();regs=I();static=I()
        api('cuOccupancyMaxActiveBlocksPerMultiprocessor',[C.POINTER(I),ptr,I,C.c_size_t],
            C.byref(active),function,128,dynamic)
        api('cuFuncGetAttribute',[C.POINTER(I),I,ptr],C.byref(regs),4,function)
        api('cuFuncGetAttribute',[C.POINTER(I),I,ptr],C.byref(static),1,function)
        result={'variant':args.variant,'parts':args.parts,'context':2064,'zero_inputs':True,
                'cubin_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'registers':regs.value,'static_shared':static.value,'dynamic_shared':dynamic,
                'driver_max_active_blocks_per_sm':active.value,'no_torch_or_jit_imports':True}
        with args.output.open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:
        api('cuCtxDestroy_v2',[ptr],ctx)


if __name__=='__main__':main()
