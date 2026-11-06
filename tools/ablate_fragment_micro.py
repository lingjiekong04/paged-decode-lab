"""MMA-fragment microkernel experiment; never an engine/attention benchmark.

Both variants include QK, row max/exp, ordered denominator and a PV anchor.
Timing is a CUDA Graph device interval per harness call, not denominator-only time.
"""
import argparse,csv,hashlib,json,os,random,re,subprocess,tempfile,time
from pathlib import Path
from _common import ROOT,write_json,fingerprint
from fragment_asm import coordinates,marker_asm,shared_ordered_asm
from cubin_text import fingerprint as text_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    def query(fields,kind='gpu'):
        return subprocess.check_output(['nvidia-smi',f'--query-{kind}='+fields,'--format=csv,noheader'],text=True)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    uuid=next(u for i,u in [list(map(str.strip,s.split(','))) for s in query('index,uuid').splitlines()] if i==visible or u==visible)
    def pids():return {int(p) for line in query('pid,gpu_uuid','compute-apps').splitlines() for p,u in [list(map(str.strip,line.split(',')))] if u==uuid}
    assert not pids(),'selected GPU busy'
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability(0)==(8,0)
    torch.empty(1,device='cuda');torch.cuda.synchronize();own=pids();assert len(own)==1
    def guard():
        assert pids()==own,'GPU sharing detected: discard attempt'
        return query('index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw')
    from paged_decode.experimental_fragment_micro import fragment_micro
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{'gpu_uuid':uuid,'host_pids':sorted(own),'seed':20261104,'repeats':5,
        'unrolls':[32,128],'replays':20,'num_warps':4,'fp_fusion':False,
        'scope':__doc__,'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/'tools/fragment_asm.py',ROOT/'src/paged_decode/experimental_fragment_micro.py')}})
    all_checks=[];all_timings=[];all_resources=[]
    for d,tile in ((64,256),(128,128)):
        guard();print('CASE',d,tile,flush=True)
        folder=args.output/f'd{d}-t{tile}';folder.mkdir()
        torch.manual_seed(20261104)
        q=torch.randn(16,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(d,tile,device='cuda',dtype=torch.bfloat16);v=torch.randn(tile,d,device='cuda',dtype=torch.bfloat16)
        initial=torch.rand(16,4,device='cuda');denom=torch.empty(16,device='cuda');pv=torch.empty(16,d,device='cuda')
        tags=torch.empty(16,tile,device='cuda',dtype=torch.int32);prob=torch.empty(16,tile,device='cuda')
        length=torch.tensor([tile],device='cuda',dtype=torch.int32)
        tensors=(q,k,v,initial,denom,pv,tags,prob,length)
        options={};compiled={}
        for mode in ('map','baseline','shared'):
            asm,cons,pack=shared_ordered_asm(tile) if mode=='shared' else marker_asm(tile)
            options[mode]=dict(D=d,TILE=tile,MODE=mode,ASM=asm,CONSTRAINTS=cons,PACK=pack,num_warps=4,enable_fp_fusion=False)
        def call(mode):return fragment_micro[(1,)](*tensors,**options[mode])
        mapped=call('map');torch.cuda.synchronize()
        actual=tags.cpu().tolist()
        assert len({x for row in actual for x in row})==16*tile
        for thread in range(128):
            for slot in range(tile//8):
                row,col=coordinates(thread,slot,tile)
                assert actual[row][col]==thread*(tile//8)+slot,'GPU mapping gate failed'
        (folder/'ownership.csv').write_text('row,column,thread,warp,lane,pack_slot\n'+''.join(
            f'{r},{c},{actual[r][c]//(tile//8)},{actual[r][c]//(tile//8)//32},{actual[r][c]//(tile//8)%32},{actual[r][c]%(tile//8)}\n' for r in range(16) for c in range(tile)))
        # Compile and inspect before allowing the hard-coded asm to execute.
        for mode in ('baseline','shared'):
            kernel=fragment_micro.warmup(*tensors,grid=(1,),**options[mode]);compiled[mode]=kernel
            ir=kernel.asm['ttgir'];layout=re.search(r'^#mma = (.*)$',ir,re.M)
            assert layout and 'versionMajor = 2' in layout[1] and 'warpsPerCTA = [1, 4]' in layout[1]
            if mode=='shared':
                calls=[line for line in ir.splitlines() if 'elementwise_inline_asm' in line]
                assert len(calls)==1 and calls[0].count('tensor<16x'+str(tile)+'xf32, #mma>')==3
            # Pinned Triton 3.2 initializes driver resource fields lazily.
            # Loading a compiled function does not execute the candidate kernel.
            kernel._init_handles()
            for ext in ('ttir','ttgir','llir','ptx','cubin'):
                value=kernel.asm[ext];(folder/(mode+'.'+ext)).write_bytes(value if isinstance(value,bytes) else value.encode())
            tool='/usr/local/cuda-12.4/bin/cuobjdump'
            resource=subprocess.check_output([tool,'--dump-resource-usage',str(folder/(mode+'.cubin'))],text=True)
            sass=subprocess.check_output([tool,'--dump-sass',str(folder/(mode+'.cubin'))],text=True)
            (folder/(mode+'.resources.txt')).write_text(resource);(folder/(mode+'.sass.txt')).write_text(sass)
            row={'d':d,'tile':tile,'variant':mode,'registers':kernel.n_regs,'spills':kernel.n_spills,
                'dynamic_shared':kernel.metadata.shared,'machine_text':text_fingerprint(kernel.asm['cubin'],kernel.metadata.name),
                'static_ldl':len(re.findall(r'\bLDL\b',sass)),'static_stl':len(re.findall(r'\bSTL\b',sass)),
                'stack_bytes':int(re.search(r'STACK:(\d+)',resource)[1]),'static_shared':int(re.search(r'SHARED:(\d+)',resource)[1]),
                'ptx_shuffle_sites':sum('shfl.sync.' in line and not line.lstrip().startswith('//') for line in kernel.asm['ptx'].splitlines())}
            all_resources.append(row);print('RESOURCE',json.dumps(row),flush=True)
        # No catastrophic register spill is permitted before timing a candidate.
        assert next(r for r in all_resources if r['d']==d and r['variant']=='shared')['stack_bytes']==0
        for mode in ('baseline','shared'):call(mode)
        torch.cuda.synchronize()
        graphs={};stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for mode in ('baseline','shared'):
            with torch.cuda.stream(stream):
                for _ in range(3):call(mode)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call(mode)
            graphs[mode]=graph
        checks=[]
        lengths=[0,1,7,8,9,tile-1,tile]
        for seed in range(4):
            for n in lengths:
                torch.manual_seed(400+seed*100+n)
                q.normal_();k.normal_();v.normal_();length.fill_(n)
                if seed==0:initial.zero_()
                elif seed==1:initial.uniform_()
                elif seed==2:initial.copy_(torch.logspace(-30,20,64,device='cuda').reshape(16,4))
                else:initial.fill_(1e-40)
                if seed==3:q.zero_()  # uniform probabilities, plus subnormal initial state.
                k[:,n:]=float('nan');v[n:]=float('nan')
                call('map');torch.cuda.synchronize()
                p=prob.cpu();state=initial.cpu().clone()
                for block in range(tile//8):
                    state=state+p[:,block*8:block*8+8:2]
                    state=state+p[:,block*8+1:block*8+8:2]
                gold=(state[:,0]+state[:,2])+(state[:,1]+state[:,3])
                results={}
                for mode in ('baseline','shared'):
                    for _ in range(4):graphs[mode].replay()
                    torch.cuda.synchronize();got=denom.clone();got_pv=pv.clone()
                    call(mode);torch.cuda.synchronize()
                    results[mode]=(got,got_pv)
                    row={'d':d,'tile':tile,'seed':seed,'length':n,'variant':mode,
                        'denominator_cpu_bitwise':torch.equal(got.cpu(),gold),'graph_eager_equal':torch.equal(got,denom) and torch.equal(got_pv,pv),
                        'finite':bool(torch.isfinite(got).all() and torch.isfinite(got_pv).all()),
                        'max_abs_cpu':(got.cpu()-gold).abs().max().item()}
                    checks.append(row)
                    assert row['denominator_cpu_bitwise'] and row['graph_eager_equal'] and row['finite'],row
                assert torch.equal(results['baseline'][0],results['shared'][0]) and torch.equal(results['baseline'][1],results['shared'][1])
        write_json(folder/'correctness.json',{'status':'passed','checks':checks,'bitwise_pairs':28})
        all_checks.extend(checks);guard();print('CORRECTNESS passed',d,len(checks),flush=True)
        q.normal_();k.normal_();v.normal_();initial.uniform_();length.fill_(tile)
        timing_graphs={}
        for mode in ('baseline','shared'):
            for unroll in (32,128):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(unroll):call(mode)
                graph.replay();torch.cuda.synchronize();timing_graphs[mode,unroll]=graph
        samples=[];snapshots=[];rng=random.Random(20261104)
        for repeat in range(5):
            before=guard();order=list(timing_graphs);rng.shuffle(order)
            for ordinal,(mode,unroll) in enumerate(order):
                graph=timing_graphs[mode,unroll]
                for _ in range(3):graph.replay()
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();start.record()
                for _ in range(20):graph.replay()
                submitted=time.perf_counter();end.record();end.synchronize()
                samples.append({'d':d,'tile':tile,'variant':mode,'unroll':unroll,'repeat':repeat,'order':ordinal,
                    'device_interval_us':start.elapsed_time(end)*1000/(20*unroll),'host_submit_us_per_replay':(submitted-wall)*1e6/20})
            snapshots.append({'repeat':repeat,'before':before,'after':guard()})
        write_json(folder/'timing.json',{'rows':samples,'gpu_snapshots':snapshots});all_timings.extend(samples)
        del graphs,timing_graphs,graph;torch.cuda.synchronize()
    write_json(args.output/'summary.json',{'status':'passed','checks':all_checks,'timings':all_timings,'resources':all_resources,
        'new_engine_backend':False,'scope':__doc__})


if __name__=='__main__':main()
