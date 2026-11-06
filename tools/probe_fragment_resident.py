"""Compile and byte-check the single num_stages=2 change before larger regressions."""
import argparse
from pathlib import Path
from _common import write_json,fingerprint
from profile_fragment_model_stages import ExclusiveGPU
from partition_residency_common import synthetic,Driver,resource_record
from fragment_resident_common import prepare,complete,strict_equal,VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);gpu=ExclusiveGPU(a.output)
    try:
        import torch
        from paged_decode.matched_splitkv import allocate_matched_resources
        from flash_attn import flash_attn_with_kvcache
        torch.set_num_threads(4)
        with torch.inference_mode():
            spec,tensors=synthetic(torch);q,k,v,table,lengths=tensors
            native=flash_attn_with_kvcache(q[:,None],k,v,cache_seqlens=lengths,block_table=table,causal=True).squeeze(1).clone()
            rows={};outputs={};parts={};lses={}
            for variant in VARIANTS:
                r=allocate_matched_resources(spec,'cuda',num_splits=36)
                c=prepare(*tensors,r,variant)
                outputs[variant]=complete(*tensors,r,variant).clone()
                parts[variant]=r.partials.clone();lses[variant]=r.lse.clone()
                assert strict_equal(torch,outputs[variant],native)
                rows[variant]=resource_record(c,Driver())
                for ext in ('ttgir','ptx','cubin'):
                    data=c.asm[ext];(a.output/(variant+'.'+ext)).write_bytes(data if isinstance(data,bytes) else data.encode())
            for v in VARIANTS:
                assert strict_equal(torch,parts[v],parts['original']) and strict_equal(torch,lses[v],lses['original'])
            gpu.guard()
            write_json(a.output/'summary.json',{'status':'passed','resources':rows,'output_partials_lse_bytes_equal':True,'environment':fingerprint()})
            print(rows,flush=True)
    finally:gpu.close()


if __name__=='__main__':main()
