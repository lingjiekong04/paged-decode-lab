"""Compile and compare a small candidate before launching a full experiment."""
import argparse,hashlib
from pathlib import Path
from _common import require_gpu,write_json,ROOT
from late_v_ablation_common import complete,partition
from cubin_text import fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch=require_gpu();torch.set_num_threads(4)
    from paged_decode.spec import DecodeSpec
    from paged_decode.matched_splitkv import allocate_matched_resources
    from paged_decode.torch_reference import dense_reference
    rows=[]
    for d in (64,128,256):
        torch.manual_seed(123);spec=DecodeSpec(1,32,8,d,2051)
        q=torch.randn(1,32,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(spec.table_width,256,8,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.randperm(spec.table_width,device='cuda').to(torch.int32)[None]
        lens=torch.tensor([1025],device='cuda',dtype=torch.int32);inputs=(q,k,v,table,lens)
        resources={v:allocate_matched_resources(spec,'cuda') for v in ('baseline','late_v')}
        outputs={v:complete(*inputs,resources[v],v).clone() for v in resources}
        gold=dense_reference(*inputs)
        for out in outputs.values():torch.testing.assert_close(out.float(),gold,atol=.02,rtol=.02)
        row={'dim':d,'output_equal':torch.equal(outputs['baseline'],outputs['late_v']),
            'partials_equal':torch.equal(resources['baseline'].partials,resources['late_v'].partials),
            'lse_equal':torch.equal(resources['baseline'].lse,resources['late_v'].lse)}
        compiled={}
        for variant in resources:
            kernel=partition(*inputs,resources[variant],variant)
            compiled[variant]={'code':fingerprint(kernel.asm['cubin'],kernel.metadata.name),
                'registers':kernel.n_regs,'spill_field':kernel.n_spills,'shared_bytes':kernel.metadata.shared}
        row['compiled']=compiled
        row['machine_code_equal']=compiled['baseline']['code']==compiled['late_v']['code']
        print(row,flush=True);rows.append(row)
        assert all(row[k] for k in ('output_equal','partials_equal','lse_equal')),row
    write_json(args.output,{'status':'passed','rows':rows,
        'candidate_sha256':hashlib.sha256((ROOT/'src/paged_decode/experimental_late_v.py').read_bytes()).hexdigest()})


if __name__=='__main__':main()
