"""Validate four-lane states across repeated tile/alpha updates; no timing."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
from _common import ROOT,write_json,fingerprint
from fragment_asm import shared_ordered_asm,marker_asm,coordinates


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def query(fields,kind='gpu'):return subprocess.check_output(['nvidia-smi',f'--query-{kind}='+fields,'--format=csv,noheader'],text=True)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    uuid=next(u for i,u in [list(map(str.strip,s.split(','))) for s in query('index,uuid').splitlines()] if i==visible or u==visible)
    def pids():return {int(p) for line in query('pid,gpu_uuid','compute-apps').splitlines() for p,u in [list(map(str.strip,line.split(',')))] if u==uuid}
    assert not pids(),'selected GPU busy'
    import torch
    torch.set_num_threads(4);assert torch.cuda.get_device_capability(0)==(8,0)
    torch.empty(1,device='cuda');torch.cuda.synchronize();own=pids();assert len(own)==1
    def guard():assert pids()==own,'GPU sharing detected'
    from paged_decode.experimental_fragment_micro import fragment_lane_trace,fragment_micro
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{'gpu_uuid':uuid,'host_pids':sorted(own),'scope':__doc__,
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/'tools/fragment_asm.py',ROOT/'src/paged_decode/experimental_fragment_micro.py')}})
    checks=[];resources=[]
    for d,tile in ((64,256),(128,128)):
        guard();folder=args.output/f'd{d}-t{tile}';folder.mkdir();torch.manual_seed(20261104)
        q=torch.randn(16,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(d,tile,device='cuda',dtype=torch.bfloat16)
        v=torch.randn(tile,d,device='cuda',dtype=torch.bfloat16);length=torch.tensor([tile],device='cuda',dtype=torch.int32)
        init={m:torch.zeros(16,4,device='cuda') for m in ('baseline','shared')}
        state={m:torch.empty_like(init[m]) for m in init};pv={m:torch.empty(16,d,device='cuda') for m in init}
        prob={m:torch.empty(16,tile,device='cuda') for m in init}
        # Re-probe ownership in this process before compiling/launching state asm.
        tags=torch.empty(16,tile,device='cuda',dtype=torch.int32);denom=torch.empty(16,device='cuda')
        asm,cons,pack=marker_asm(tile)
        mapped=fragment_micro[(1,)](q,k,v,init['baseline'],denom,pv['baseline'],tags,prob['baseline'],length,
            D=d,TILE=tile,MODE='map',ASM=asm,CONSTRAINTS=cons,PACK=pack,num_warps=4,enable_fp_fusion=False)
        actual=tags.cpu().tolist()
        for t in range(128):
            for s in range(pack):
                r,c=coordinates(t,s,tile);assert actual[r][c]==t*pack+s
        opts={};inputs={}
        for mode in init:
            asm,cons,pack=shared_ordered_asm(tile,states=True)
            opts[mode]=dict(D=d,TILE=tile,MODE=mode,ASM=asm,CONSTRAINTS=cons,PACK=pack,num_warps=4,enable_fp_fusion=False)
            inputs[mode]=(q,k,v,init[mode],state[mode],pv[mode],prob[mode],length)
            kernel=fragment_lane_trace.warmup(*inputs[mode],grid=(1,),**opts[mode])
            layout=re.search(r'^#mma = (.*)$',kernel.asm['ttgir'],re.M)
            assert layout and 'versionMajor = 2' in layout[1] and 'warpsPerCTA = [1, 4]' in layout[1]
            if mode=='shared':
                calls=[s for s in kernel.asm['ttgir'].splitlines() if 'elementwise_inline_asm' in s]
                assert len(calls)==1 and calls[0].count('tensor<16x'+str(tile)+'xf32, #mma>')==3
            kernel._init_handles()
            for ext in ('ttgir','ptx','cubin'):
                value=kernel.asm[ext];(folder/(mode+'.'+ext)).write_bytes(value if isinstance(value,bytes) else value.encode())
            resources.append({'d':d,'variant':mode,'registers':kernel.n_regs,'spills':kernel.n_spills,'dynamic_shared':kernel.metadata.shared})
        def call(mode):return fragment_lane_trace[(1,)](*inputs[mode],**opts[mode])
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graphs={}
        for mode in init:
            with torch.cuda.stream(stream):
                for _ in range(3):call(mode)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call(mode)
            graphs[mode]=graph
        for trial in range(4):
            torch.manual_seed(17+trial)
            gold=(torch.zeros(16,4) if trial==0 else torch.rand(16,4)*(1e20 if trial==2 else 1.))
            previous={m:gold.cuda() for m in init}
            for step,n in enumerate((tile,1,0,7,8,9,tile-1)):
                q.normal_();k.normal_();v.normal_();length.fill_(n)
                k[:,n:]=float('nan');v[n:]=float('nan')
                alpha=torch.rand(16,1,device='cuda')
                gold=gold*alpha.cpu()
                for mode in init:init[mode].copy_(previous[mode]*alpha)
                for mode in init:
                    for _ in range(4):graphs[mode].replay()
                torch.cuda.synchronize()
                p=prob['baseline'].cpu()
                for block in range(tile//8):
                    gold=gold+p[:,8*block:8*block+8:2]
                    gold=gold+p[:,8*block+1:8*block+8:2]
                for mode in init:
                    got=state[mode].clone();gotpv=pv[mode].clone();call(mode);torch.cuda.synchronize()
                    row={'d':d,'trial':trial,'step':step,'length':n,'variant':mode,'four_states_cpu_bitwise':torch.equal(got.cpu(),gold),
                        'graph_eager_equal':torch.equal(got,state[mode]) and torch.equal(gotpv,pv[mode]),'finite':bool(torch.isfinite(got).all() and torch.isfinite(gotpv).all())}
                    assert all(row[k] for k in ('four_states_cpu_bitwise','graph_eager_equal','finite')),row
                    checks.append(row);previous[mode]=got
                assert torch.equal(state['baseline'],state['shared']) and torch.equal(pv['baseline'],pv['shared']) and torch.equal(prob['baseline'],prob['shared'])
            guard()
        print('RECURRENCE passed',d,flush=True)
        del graphs,graph;torch.cuda.synchronize()
    write_json(args.output/'summary.json',{'status':'passed','checks':checks,'resources':resources,'bitwise_pairs':56,'new_engine_backend':False})


if __name__=='__main__':main()
