"""CPU-only metadata and cached workspace allocation costs, outside GPU timing."""
import argparse,time,hashlib
from pathlib import Path
from _common import require_gpu,fingerprint,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch=require_gpu(); torch.set_num_threads(4)
    from paged_decode.spec import DecodeSpec
    from paged_decode.runtime import _validate
    from paged_decode.matched_splitkv import allocate_matched_resources
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    rows=[]
    for heads,kv in ((8,2),(16,4)):
        for n in (512,8192):
            spec=DecodeSpec(1,heads,kv,256,n)
            r=allocate_matched_resources(spec,'cuda')
            q=torch.empty(1,heads,256,device='cuda',dtype=torch.bfloat16)
            k=torch.empty(spec.table_width,256,kv,256,device='cuda',dtype=torch.bfloat16);v=torch.empty_like(k)
            table=torch.empty(1,spec.table_width,device='cuda',dtype=torch.int32)
            lens=torch.empty(1,device='cuda',dtype=torch.int32)
            for name,f in [('validate_only',lambda:_validate(q,k,v,table,lens,r,'triton_base',.0625)),
                           ('allocate_only',lambda:allocate_matched_resources(spec,'cuda'))]:
                for _ in range(20): f()
                for repeat in range(5):
                    torch.cuda.synchronize()
                    begin=time.perf_counter()
                    for _ in range(1000): f()
                    elapsed=time.perf_counter()-begin
                    rows.append({'query_heads':heads,'context':n,'variant':name,'repeat':repeat,'cpu_wall_us':elapsed*1e3})
    write_json(args.output/'summary.json',{'rows':rows,'iterations':1000,
        'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'note':'Warm cached allocations; synchronization before timer only. No kernel launches. Metadata tensor contents are unused.'})


if __name__=='__main__':main()
