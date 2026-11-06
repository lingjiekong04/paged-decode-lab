"""Fixed unroll128 primary, unroll convergence, randomized five-round operator matrix."""
import argparse,hashlib,itertools,os,random,subprocess,time
from pathlib import Path
from dataclasses import asdict
from _common import require_gpu,fingerprint,write_json
from paged_decode.benchmark_protocol import UNROLLS,interval_per_operator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',choices=('small','large','standard'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--replays',type=int,default=20)
    args=p.parse_args()
    if min(args.repeats,args.replays)<1:p.error('positive repeats/replays required')
    def query(arguments):
        result=subprocess.run(['nvidia-smi',*arguments],capture_output=True,text=True,timeout=20)
        if result.returncode:raise RuntimeError(result.stderr)
        return result.stdout
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    devices=[list(map(str.strip,line.split(','))) for line in query(['--query-gpu=index,uuid','--format=csv,noheader']).splitlines()]
    selected_uuid=next(row[1] for row in devices if (row[0]==visible if visible.isdigit() else row[1].startswith(visible)))
    def active_gpu_pids(apps=None):
        apps=query(['--query-compute-apps=pid,gpu_uuid','--format=csv,noheader']) if apps is None else apps
        return {int(parts[0]) for line in apps.splitlines() if len(parts:=[v.strip() for v in line.split(',')])>=2 and parts[1]==selected_uuid}
    if active_gpu_pids():raise RuntimeError('Selected GPU has resident compute processes; no benchmark launched')
    torch=require_gpu();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    probe=torch.empty(1,device='cuda');torch.cuda.synchronize()
    host_pids=active_gpu_pids()
    if len(host_pids)!=1:raise RuntimeError('Cannot establish exclusive GPU process identity')
    host_pid=next(iter(host_pids))
    from flash_attn import flash_attn_with_kvcache
    from paged_decode.spec import DecodeSpec
    from paged_decode.runtime import allocate_resources,paged_decode
    from paged_decode.matched_splitkv import allocate_matched_resources,matched_decode
    from paged_decode.torch_reference import dense_reference
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{**vars(args),'output':str(args.output),'unrolls':UNROLLS,
        'primary_unroll':128,'convergence_limit':.05,'seed':20261103,'torch_threads':4,
        'pid':os.getpid(),'host_gpu_pid':host_pid,'gpu_uuid':selected_uuid,
        'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def gpu_snapshot():
        return {key:subprocess.run(['nvidia-smi',*opts],capture_output=True,text=True,timeout=20).stdout
            for key,opts in [('devices',['--query-gpu=index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader']),
                             ('apps',['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])]}
    hq,hkv,dims=(8,2,(256,)) if args.geometry=='small' else (16,4,(256,)) if args.geometry=='large' else (32,8,(64,128))
    shapes=list(itertools.product((1,8,32),(512,2048,8192),dims))
    rng=random.Random(20261103);rng.shuffle(shapes)
    rows=[];hostrows=[];checks=[]
    for b,n,d in shapes:
        print('SHAPE',args.geometry,b,n,d,flush=True)
        torch.manual_seed(123)
        spec=DecodeSpec(b,hq,hkv,d,n)
        q=torch.randn(b,hq,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(b*spec.table_width,256,hkv,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.randperm(b*spec.table_width,device='cuda').int().view(b,spec.table_width)
        lengths=torch.full((b,),n,device='cuda',dtype=torch.int32)
        resources={'triton_tc_splitk':allocate_resources(spec,'cuda','triton_tc_splitk'),
                   'triton_matched':allocate_matched_resources(spec,'cuda')}
        def call(name,allocate=False):
            if name=='flash_attn':return flash_attn_with_kvcache(q[:,None],k,v,block_table=table,cache_seqlens=lengths,causal=True).squeeze(1)
            r=(allocate_matched_resources(spec,'cuda') if name=='triton_matched' else allocate_resources(spec,'cuda',name)) if allocate else resources[name]
            return matched_decode(q,k,v,table,lengths,resources=r) if name=='triton_matched' else paged_decode(q,k,v,table,lengths,resources=r,backend=name)
        backends=('flash_attn','triton_tc_splitk','triton_matched')
        gold=dense_reference(q,k,v,table,lengths)
        native=call('flash_attn').clone()
        for name in backends:
            for _ in range(10):actual=call(name)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual.float(),gold,atol=.02,rtol=.02)
            diff=actual.float()-gold
            checks.append({'batch':b,'context':n,'dim':d,'backend':name,
                'max_abs_fp32':diff.abs().max().item(),'rmse_fp32':diff.square().mean().sqrt().item(),
                'native_equal':torch.equal(actual,native),
                'plan':asdict(resources[name].plan) if name=='triton_matched' else None})
        graphs={};outputs={}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for name in backends:
            for unroll in UNROLLS:
                with torch.cuda.stream(stream):
                    for _ in range(3):call(name)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(unroll):captured=call(name)
                graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(captured.float(),gold,atol=.02,rtol=.02)
                graphs[name,unroll]=graph;outputs[name,unroll]=captured
        snapshots=[]
        for repeat in range(args.repeats):
            before=gpu_snapshot()
            if active_gpu_pids(before['apps'])!={host_pid}:raise RuntimeError('GPU sharing detected; discard this attempt for formal timing')
            snapshots.append(before)
            order=list(graphs);rng.shuffle(order)
            for order_index,(name,unroll) in enumerate(order):
                graph=graphs[name,unroll]
                for _ in range(3):graph.replay()
                torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin=time.perf_counter();start.record()
                for _ in range(args.replays):graph.replay()
                submitted=time.perf_counter();end.record();end.synchronize()
                row={'batch':b,'context':n,'dim':d,'backend':name,'repeat':repeat,'order_index':order_index,'unroll':unroll,
                    'device_interval_us':interval_per_operator(start.elapsed_time(end),args.replays,unroll),
                    'host_submit_us_per_replay':(submitted-begin)*1e6/args.replays,
                    'wall_us_per_operator':(time.perf_counter()-begin)*1e6/(args.replays*unroll),
                    'workspace_bytes':(resources[name].partials.numel()+resources[name].lse.numel())*4 if name=='triton_matched' else spec.split_workspace_bytes if name=='triton_tc_splitk' else None}
                rows.append(row)
            order=list(backends);rng.shuffle(order)
            for name in order:
                samples={}
                for allocation in (False,True):
                    values=[]
                    for _ in range(20):
                        torch.cuda.synchronize();begin=time.perf_counter();call(name,allocation);torch.cuda.synchronize()
                        values.append((time.perf_counter()-begin)*1e6)
                    samples['allocation' if allocation else 'preallocated']=values
                hostrows.append({'batch':b,'context':n,'dim':d,'backend':name,'repeat':repeat,**samples})
            after=gpu_snapshot();before['after']=after
            if active_gpu_pids(after['apps'])!={host_pid}:raise RuntimeError('GPU sharing detected after round; formal run stopped')
        write_json(args.output/f'b{b}-l{n}-d{d}.json',{'rows':[r for r in rows if (r['batch'],r['context'],r['dim'])==(b,n,d)],
            'host_rows':[r for r in hostrows if (r['batch'],r['context'],r['dim'])==(b,n,d)],'gpu_snapshots':snapshots})
        del graphs,outputs,graph,captured,resources,actual,gold,native,q,k,v,table,lengths
        torch.cuda.synchronize();torch.cuda.empty_cache()
    write_json(args.output/'summary.json',{'status':'completed','rows':rows,'host_rows':hostrows,'checks':checks,
        'note':'Primary fixed unroll128 per operator; execution intervals include graph-node gaps; no model throughput or pure kernel-busy claim.'})


if __name__=='__main__':main()
