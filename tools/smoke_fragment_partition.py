"""Paged multi-tile preflight and per-CTA fragment probe before broad testing."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
from _common import ROOT,write_json,fingerprint
from fragment_asm import coordinates
from fragment_partition_common import prepare,complete,strict_equal


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def query(fields,kind='gpu'):return subprocess.check_output(['nvidia-smi',f'--query-{kind}='+fields,'--format=csv,noheader'],text=True)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    uuid=next(u for i,u in [list(map(str.strip,s.split(','))) for s in query('index,uuid').splitlines()] if i==visible or u==visible)
    def pids():return {int(p) for line in query('pid,gpu_uuid','compute-apps').splitlines() for p,u in [list(map(str.strip,line.split(',')))] if u==uuid}
    assert not pids(),'selected GPU busy'
    import torch
    torch.set_num_threads(4);torch.empty(1,device='cuda');torch.cuda.synchronize();own=pids();assert len(own)==1
    from paged_decode.spec import DecodeSpec
    from paged_decode.matched_splitkv import allocate_matched_resources
    from paged_decode.torch_reference import dense_reference
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{'gpu_uuid':uuid,'host_pids':sorted(own),'source_sha256':{
        p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),ROOT/'tools/fragment_partition_common.py',ROOT/'tools/fragment_asm.py',ROOT/'tools/fragment_partition_asm.py',ROOT/'src/paged_decode/experimental_fragment_partition.py')}})
    rows=[]
    for d in (64,128):
        assert pids()==own
        torch.manual_seed(123);spec=DecodeSpec(2,4,1,d,2051)
        q=torch.randn(2,4,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(2*spec.table_width,256,1,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.randperm(2*spec.table_width,device='cuda').int().reshape(2,spec.table_width)
        lengths=torch.tensor([1025,0],device='cuda',dtype=torch.int32);tensors=(q,k,v,table,lengths)
        r={m:allocate_matched_resources(spec,'cuda',num_splits=1) for m in ('baseline','shared')}
        folder=args.output/f'd{d}';folder.mkdir();records={};outs={}
        for mode in ('baseline','shared'):
            kernel=prepare(*tensors,r[mode],mode)
            for ext in ('ttir','ttgir','llir','ptx','cubin'):
                val=kernel.asm[ext];(folder/(mode+'.'+ext)).write_bytes(val if isinstance(val,bytes) else val.encode())
            resources=subprocess.check_output(['/usr/local/cuda-12.4/bin/cuobjdump','--dump-resource-usage',str(folder/(mode+'.cubin'))],text=True)
            (folder/(mode+'.resources.txt')).write_text(resources)
            records[mode]={'registers':kernel.n_regs,'spills':kernel.n_spills,'dynamic_shared':kernel.metadata.shared,
                           'static_shared':int(re.search(r'SHARED:(\d+)',resources)[1]),'stack':int(re.search(r'STACK:(\d+)',resources)[1])}
            outs[mode]=complete(*tensors,r[mode],mode).clone();torch.cuda.synchronize()
        gold=dense_reference(*tensors)
        for out in outs.values():torch.testing.assert_close(out.float(),gold,atol=.02,rtol=.02)
        checks={'output_bits':strict_equal(torch,outs['baseline'],outs['shared']),
                'partials_bits':strict_equal(torch,r['baseline'].partials,r['shared'].partials),
                'lse_bits':strict_equal(torch,r['baseline'].lse,r['shared'].lse)}
        assert all(checks.values()),checks
        tile=r['shared'].plan.tile_tokens
        tags=torch.full((2,1,1,16,tile),-1,device='cuda',dtype=torch.int32)
        probe_r=allocate_matched_resources(spec,'cuda',num_splits=1)
        probe=prepare(*tensors,probe_r,'shared',tags=tags)
        complete(*tensors,probe_r,'shared');torch.cuda.synchronize()
        actual=tags.cpu();mismatches=0
        for thread in range(128):
            for slot in range(tile//8):
                row,col=coordinates(thread,slot,tile)
                mismatches+=int(actual[0,0,0,row,col])!=thread*(tile//8)+slot
        assert mismatches==0 and bool((actual[1]==-1).all())
        assert strict_equal(torch,probe_r.out,outs['shared'])
        for ext in ('ttgir','ptx'):(folder/('probe.'+ext)).write_bytes(probe.asm[ext].encode())
        row={'d':d,'resources':records,'mapping_mismatches':mismatches,'inactive_tags_untouched':True,**checks}
        rows.append(row);print(json.dumps(row),flush=True)
    assert pids()==own
    write_json(args.output/'summary.json',{'status':'passed','rows':rows,'scope':'Paged multiple tiles, one explicit split; no production backend change'})


if __name__=='__main__':main()
