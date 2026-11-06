"""D64/128 stage diagnosis with D256 controls; production kernels remain unchanged."""
import argparse,hashlib,json,os,random,shutil,subprocess,time
from pathlib import Path
from dataclasses import asdict
from _common import require_gpu,fingerprint,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    def query(opts):
        p=subprocess.run(['nvidia-smi',*opts],capture_output=True,text=True,timeout=20)
        if p.returncode:raise RuntimeError(p.stderr)
        return p.stdout
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    devices=[list(map(str.strip,line.split(','))) for line in query(['--query-gpu=index,uuid','--format=csv,noheader']).splitlines()]
    uuid=next(v for i,v in devices if (i==visible if visible.isdigit() else v.startswith(visible)))
    def pids():
        rows=[list(map(str.strip,line.split(','))) for line in query(['--query-compute-apps=pid,gpu_uuid','--format=csv,noheader']).splitlines()]
        return {int(r[0]) for r in rows if len(r)>=2 and r[1]==uuid}
    if pids():raise RuntimeError('selected GPU busy; no diagnosis started')
    torch=require_gpu();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.empty(1,device='cuda');torch.cuda.synchronize()
    own=pids()
    if len(own)!=1:raise RuntimeError('cannot establish exclusive host GPU PID')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    cases=[(b,n,d) for d in (64,128,256) for b,n in ((1,512),(1,8192),(32,8192))]
    random.Random(20261104).shuffle(cases)
    write_json(args.output/'arguments.json',{'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases':cases,'hq':32,'hkv':8,'repeats':5,'replays':20,'unrolls':[32,128],'primary_unroll':128,
        'seed':20261104,'gpu_uuid':uuid,'host_pids':list(own),'nsight_tools':{x:shutil.which(x) for x in ('ncu','nsys','cuobjdump')},
        'measurement':'fixed inputs; randomized variant/unroll within every round; no model throughput claim'})
    import triton
    from flash_attn import flash_attn_with_kvcache
    from paged_decode.spec import DecodeSpec
    from paged_decode.runtime import _validate
    from paged_decode.matched_splitkv import allocate_matched_resources,matched_decode,_normalized_partition,_lse_merge
    from paged_decode.torch_reference import dense_reference
    rows=[];allcases=[]
    def guard():
        apps=pids()
        if apps!=own:raise RuntimeError('GPU sharing detected; discard this attempt')
        return query(['--query-gpu=index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'])
    for b,n,d in cases:
        guard();print('CASE',b,n,d,flush=True)
        spec=DecodeSpec(b,32,8,d,n);scale=d**-.5
        torch.manual_seed(123)
        q=torch.randn(b,32,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(b*spec.table_width,256,8,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.randperm(b*spec.table_width,device='cuda').int().reshape(b,spec.table_width)
        lengths=torch.full((b,),n,device='cuda',dtype=torch.int32)
        r=allocate_matched_resources(spec,'cuda');plan=r.plan
        folder=args.output/f'b{b}-l{n}-d{d}';folder.mkdir()
        def first():
            return _normalized_partition[(b,8,plan.splits)](q,k,v,table,lengths,r.partials,r.lse,
                Q0=q.stride(0),Q1=q.stride(1),TABLE0=table.stride(0),HQ=32,HKV=8,D=d,PARTS=plan.splits,
                PART_SIZE=plan.tokens_per_split,SCALE=scale,SCALE_LOG2=scale*1.4426950408889634,TILE=plan.tile_tokens,num_warps=4)
        def merge():
            return _lse_merge[(b,32)](r.partials,r.lse,r.out,HQ=32,D=d,PARTS=plan.splits,
                PAD_PARTS=triton.next_power_of_2(plan.splits),LANES=plan.lse_lanes,num_warps=4)
        def bare():first();merge();return r.out
        def api():return matched_decode(q,k,v,table,lengths,resources=r)
        def native():return flash_attn_with_kvcache(q[:,None],k,v,block_table=table,cache_seqlens=lengths,causal=True).squeeze(1)
        variants={'native':native,'matched_api':api,'matched_bare':bare,'partition_only':first,'merge_only':merge}
        gold=dense_reference(q,k,v,table,lengths)
        actual=api().clone();reference=native().clone()
        torch.testing.assert_close(actual.float(),gold,atol=.02,rtol=.02)
        torch.testing.assert_close(reference.float(),gold,atol=.02,rtol=.02)
        torch.testing.assert_close(bare(),actual,atol=0,rtol=0)
        compiled={}
        for label,kernel in [('partition',first()),('merge',merge())]:
            compiled[label]={'registers_per_thread':getattr(kernel,'n_regs',None),'spill_count':getattr(kernel,'n_spills',None),
                'metadata':str(kernel.metadata),'ptx_file':f'{label}.ptx'}
            (folder/f'{label}.ptx').write_bytes(kernel.asm['ptx'].encode())
        case={'batch':b,'context':n,'dim':d,'plan':asdict(plan),'native_equal':torch.equal(actual,reference),
            'max_abs_fp32':(actual.float()-gold).abs().max().item(),'rmse_fp32':(actual.float()-gold).square().mean().sqrt().item(),
            'partition_programs':b*8*plan.splits,'merge_programs':b*32,'query_rows_used':4,'query_rows_computed':16,
            'workspace_bytes':(r.partials.numel()+r.lse.numel())*4,'compiled':compiled}
        write_json(folder/'case.json',case);allcases.append(case)
        for f in variants.values():
            for _ in range(10):f()
        graphs={};stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for name,f in variants.items():
            for unroll in (32,128):
                with torch.cuda.stream(stream):
                    for _ in range(3):f()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(unroll):captured=f()
                graph.replay();torch.cuda.synchronize()
                if name in ('native','matched_api','matched_bare'):
                    torch.testing.assert_close(captured.float(),gold,atol=.02,rtol=.02)
                graphs[name,unroll]=graph
        rng=random.Random(20261104);samples=[];host=[];snapshots=[]
        for repeat in range(5):
            before=guard();order=list(graphs);rng.shuffle(order)
            for index,(name,unroll) in enumerate(order):
                graph=graphs[name,unroll]
                for _ in range(3):graph.replay()
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();start.record()
                for _ in range(20):graph.replay()
                submitted=time.perf_counter();end.record();end.synchronize()
                samples.append({'repeat':repeat,'order':index,'variant':name,'unroll':unroll,
                    'device_interval_us':start.elapsed_time(end)*1000/(20*unroll),
                    'host_submit_us_per_replay':(submitted-wall)*1e6/20})
            functions={'native':native,'matched_preallocated':api,'bare_preallocated':bare,
                'matched_allocate':lambda:matched_decode(q,k,v,table,lengths,resources=allocate_matched_resources(spec,'cuda'))}
            order=list(functions);rng.shuffle(order)
            for name in order:
                vals=[]
                for _ in range(20):
                    torch.cuda.synchronize();t=time.perf_counter();functions[name]();torch.cuda.synchronize();vals.append((time.perf_counter()-t)*1e6)
                host.append({'repeat':repeat,'variant':name,'synchronized_wall_us':vals})
            snapshots.append({'repeat':repeat,'before':before,'after':guard()})
        write_json(folder/'timing.json',{'rows':samples,'host_rows':host,'gpu_snapshots':snapshots})
        for name in ('native','matched_api'):
            guard()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    with torch.profiler.record_function(f'{name}-graph32'):graphs[name,32].replay()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(folder/f'{name}-trace.json'))
            guard()
        rows.extend([{**x,'batch':b,'context':n,'dim':d} for x in samples])
        write_json(folder/'completed.json',{'status':'completed'})
        del graphs,graph,captured,functions,q,k,v,table,lengths,r,gold,actual,reference
        torch.cuda.synchronize();torch.cuda.empty_cache()
    write_json(args.output/'summary.json',{'status':'completed','cases':allcases,'rows':rows,
        'notes':['No kernel changes. Static compiled resources are not achieved hardware occupancy.',
                 'Isolated stage timings cannot be subtracted mechanically from a complete call.',
                 'Graph repeats reuse identical data; profiler is separate from formal timing.']})


if __name__=='__main__':main()
