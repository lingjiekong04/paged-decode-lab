"""Isolate FP contraction on the failed MHA fixture; not a timing campaign."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
from _common import ROOT,write_json,fingerprint
from fragment_partition_common import prepare,complete,strict_equal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def query(fields,kind='gpu'):return subprocess.check_output(['nvidia-smi',f'--query-{kind}='+fields,'--format=csv,noheader'],text=True)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    uuid=next(u for i,u in [list(map(str.strip,x.split(','))) for x in query('index,uuid').splitlines()] if i==visible or u==visible)
    def pids():return {int(p) for x in query('pid,gpu_uuid','compute-apps').splitlines() for p,u in [list(map(str.strip,x.split(',')))] if u==uuid}
    assert not pids()
    import torch
    torch.set_num_threads(4);torch.empty(1,device='cuda');torch.cuda.synchronize();own=pids();assert len(own)==1
    from paged_decode.spec import DecodeSpec
    from paged_decode.matched_splitkv import allocate_matched_resources
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),ROOT/'tools/fragment_partition_common.py',ROOT/'tools/fragment_partition_asm.py',
         ROOT/'tools/fragment_asm.py',ROOT/'src/paged_decode/experimental_fragment_partition.py')}})
    torch.manual_seed(123);spec=DecodeSpec(8,8,8,64,2051)
    q=torch.randn(8,8,64,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(8*spec.table_width,256,8,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    table=torch.randperm(8*spec.table_width,device='cuda').int().reshape(8,spec.table_width)
    values=[0,1,255,256,257,1023,1024,1025];lens=torch.tensor(values,device='cuda',dtype=torch.int32)
    original=table.clone()
    for row,n in enumerate(values):
        table[row,(n+255)//256:]=-1
        if n%256:
            page=int(original[row,n//256]);k[page,n%256:]=float('nan');v[page,n%256:]=float('nan')
    inputs=(q,k,v,table,lens);rows=[]
    for fusion in (True,False):
        resources={mode:allocate_matched_resources(spec,'cuda') for mode in ('baseline','shared')}
        outputs={}
        for mode in resources:
            name=f'{mode}-fusion{fusion}';kernel=prepare(*inputs,resources[mode],mode,fp_fusion=fusion)
            for ext in ('ttgir','llir','ptx','cubin'):
                val=kernel.asm[ext];(args.output/(name+'.'+ext)).write_bytes(val if isinstance(val,bytes) else val.encode())
            sass=subprocess.check_output(['/usr/local/cuda-12.4/bin/cuobjdump','--dump-sass',str(args.output/(name+'.cubin'))],text=True)
            (args.output/(name+'.sass.txt')).write_text(sass)
            outputs[mode]=complete(*inputs,resources[mode],mode).clone();torch.cuda.synchronize()
        a,b=resources['baseline'],resources['shared']
        delta=(a.partials-b.partials).abs();idx=torch.nonzero(delta>0)
        examples=[]
        for loc in idx[:12].cpu().tolist():
            at=tuple(loc);examples.append({'index':loc,'baseline':a.partials[at].item(),'shared':b.partials[at].item()})
        row={'fp_fusion':fusion,'output_bits':strict_equal(torch,outputs['baseline'],outputs['shared']),
             'partials_bits':strict_equal(torch,a.partials,b.partials),'lse_bits':strict_equal(torch,a.lse,b.lse),
             'nonzero_partials_differences':int(idx.shape[0]),'max_abs_partials':delta.max().item(),'examples':examples}
        rows.append(row);print(json.dumps(row),flush=True)
    assert pids()==own;write_json(args.output/'summary.json',{'rows':rows,'scope':__doc__})


if __name__=='__main__':main()
