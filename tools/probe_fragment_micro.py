"""Probe register ownership before permitting the hard-coded shared microkernel."""
import argparse,csv,hashlib,json,os,re,subprocess,traceback
from pathlib import Path
from _common import ROOT,write_json,fingerprint
from fragment_asm import coordinates,marker_asm,shared_ordered_asm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--map-only',action='store_true')
    args=parser.parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    def query(fields,kind='gpu'):
        return subprocess.check_output(['nvidia-smi',f'--query-{kind}='+fields,'--format=csv,noheader'],text=True)
    devices=[list(map(str.strip,s.split(','))) for s in query('index,uuid').splitlines()]
    uuid=next(u for i,u in devices if i==visible or u==visible)
    def pids():
        return {int(p) for line in query('pid,gpu_uuid','compute-apps').splitlines() for p,u in [list(map(str.strip,line.split(',')))] if u==uuid}
    assert not pids(),'selected GPU busy; no jobs will be interrupted'
    import torch
    assert torch.cuda.get_device_capability(0)==(8,0),'only the audited sm80 geometry is supported'
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.empty(1,device='cuda');torch.cuda.synchronize();own=pids()
    assert len(own)==1,'cannot establish own host GPU PID'
    def guard():assert pids()==own,'GPU sharing detected'
    from paged_decode.experimental_fragment_micro import fragment_micro
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{'map_only':args.map_only,'gpu_uuid':uuid,'host_pids':sorted(own),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'tools/fragment_asm.py',ROOT/'src/paged_decode/experimental_fragment_micro.py']}})
    results=[]
    for d,tile in ((64,256),(128,128)):
        guard();folder=args.output/f'd{d}-t{tile}';folder.mkdir()
        torch.manual_seed(117)
        q=torch.randn(16,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(d,tile,device='cuda',dtype=torch.bfloat16)
        v=torch.randn(tile,d,device='cuda',dtype=torch.bfloat16)
        initial=torch.rand(16,4,device='cuda');denom=torch.empty(16,device='cuda')
        pv=torch.empty(16,d,device='cuda');tags=torch.empty(16,tile,device='cuda',dtype=torch.int32)
        prob=torch.empty(16,tile,device='cuda');lengths=torch.tensor([tile],device='cuda',dtype=torch.int32)
        inputs=(q,k,v,initial,denom,pv,tags,prob,lengths)
        def launch(mode):
            asm,constraints,pack=shared_ordered_asm(tile) if mode=='shared' else marker_asm(tile)
            options=dict(D=d,TILE=tile,MODE=mode,ASM=asm,CONSTRAINTS=constraints,PACK=pack,num_warps=4,enable_fp_fusion=False)
            if mode=='shared':
                compiled=fragment_micro.warmup(*inputs,grid=(1,),**options)
                text=compiled.asm['ttgir']
                layout=re.search(r'^#mma = (.*)$',text,re.M)
                assert layout and 'versionMajor = 2' in layout[1] and 'warpsPerCTA = [1, 4]' in layout[1]
                calls=[line for line in text.splitlines() if 'elementwise_inline_asm' in line]
                assert len(calls)==1 and calls[0].count('tensor<16x'+str(tile)+'xf32, #mma>')==3,'inline asm must keep both inputs and output in the audited MMA layout'
                assert compiled.metadata.num_warps==4
            return fragment_micro[(1,)](*inputs,**options)
        def export(kernel,name):
            for ext in ('ttir','ttgir','llir','ptx','cubin'):
                data=kernel.asm[ext];(folder/(name+'.'+ext)).write_bytes(data if isinstance(data,bytes) else data.encode())
            write_json(folder/(name+'.metadata.json'),{'metadata':str(kernel.metadata),'registers':kernel.n_regs,'spills':kernel.n_spills,'shared':kernel.metadata.shared})
        mapped=launch('map');torch.cuda.synchronize();export(mapped,'map')
        actual=tags.cpu().tolist();pack=tile//8
        with (folder/'ownership.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['row','column','thread','warp','lane','pack_slot'])
            for row in range(16):
                for col in range(tile):
                    thread,slot=divmod(actual[row][col],pack);writer.writerow([row,col,thread,thread//32,thread%32,slot])
        mismatch=[]
        for thread in range(128):
            for slot in range(pack):
                row,col=coordinates(thread,slot,tile)
                if actual[row][col]!=thread*pack+slot:mismatch.append([row,col,thread,slot,actual[row][col]])
        layout=re.search(r'^#mma = (.*)$',mapped.asm['ttgir'],re.M)
        row={'d':d,'tile':tile,'mismatches':len(mismatch),'mismatch_examples':mismatch[:8],
             'mma_layout':layout[1] if layout else None,'logical_elements':16*tile,'observed_unique_tags':len({x for row in actual for x in row})}
        write_json(folder/'mapping.json',row)
        print('MAPPING',json.dumps(row),flush=True)
        assert not mismatch,'register mapping differs; shared candidate is forbidden'
        assert layout and 'warpsPerCTA = [1, 4]' in layout[1]
        assert row['observed_unique_tags']==16*tile,'duplicate/missing fragment slots'
        if not args.map_only:
            baseline=launch('baseline');torch.cuda.synchronize();export(baseline,'baseline')
            expected=denom.clone();expected_pv=pv.clone()
            shared=launch('shared');torch.cuda.synchronize();export(shared,'shared')
            candidate=denom.clone()
            row.update(denominator_bitwise_equal=torch.equal(expected,candidate),pv_bitwise_equal=torch.equal(expected_pv,pv),
                       finite=bool(torch.isfinite(candidate).all()),max_abs=(expected-candidate).abs().max().item())
            print('CANDIDATE',json.dumps(row),flush=True)
            assert row['denominator_bitwise_equal'] and row['pv_bitwise_equal'] and row['finite']
        results.append(row);guard()
    write_json(args.output/'summary.json',{'status':'passed','results':results,'gpu_uuid':uuid,'new_engine_backend':False})


if __name__=='__main__':main()
