"""Controlled host-submission versus device-stage diagnosis; no kernel changes."""
import argparse,hashlib,json,random,time
from pathlib import Path
from dataclasses import asdict
from _common import require_gpu,fingerprint,write_json,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch=require_gpu()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    import triton
    from flash_attn import flash_attn_with_kvcache
    from paged_decode.spec import DecodeSpec
    from paged_decode.runtime import _validate
    from paged_decode.matched_splitkv import allocate_matched_resources,matched_decode,_normalized_partition,_lse_merge
    from paged_decode.torch_reference import dense_reference
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'repeats':5,'replays':50,'unroll':[1,8,32],'case_order':'small512,large512,small8192,large8192',
        'order':'seed20261103 randomized variant order inside every repeat','torch_threads':4})
    allrows=[]; cases=[]
    for label,length in [('small',512),('large',512),('small',8192),('large',8192)]:
        heads,kv=(8,2) if label=='small' else (16,4)
        spec=DecodeSpec(1,heads,kv,256,length)
        torch.manual_seed(123)
        q=torch.randn(1,heads,256,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(spec.table_width,256,kv,256,device='cuda',dtype=torch.bfloat16); v=torch.randn_like(k)
        table=torch.randperm(spec.table_width,device='cuda').int()[None]
        lens=torch.full((1,),length,device='cuda',dtype=torch.int32)
        r=allocate_matched_resources(spec,'cuda'); plan=r.plan; scale=256**-.5
        def first():
            _normalized_partition[(1,kv,plan.splits)](q,k,v,table,lens,r.partials,r.lse,
                Q0=q.stride(0),Q1=q.stride(1),TABLE0=table.stride(0),HQ=heads,HKV=kv,D=256,
                PARTS=plan.splits,PART_SIZE=plan.tokens_per_split,SCALE=scale,
                SCALE_LOG2=scale*1.4426950408889634,TILE=plan.tile_tokens,num_warps=4)
        def merge():
            _lse_merge[(1,heads)](r.partials,r.lse,r.out,HQ=heads,D=256,PARTS=plan.splits,
                PAD_PARTS=triton.next_power_of_2(plan.splits),LANES=plan.lse_lanes,num_warps=4)
        def bare(): first(); merge(); return r.out
        def api(): return matched_decode(q,k,v,table,lens,resources=r)
        def native():
            return flash_attn_with_kvcache(q[:,None],k,v,block_table=table,cache_seqlens=lens,causal=True).squeeze(1)
        variants={'native':native,'matched_api':api,'matched_bare':bare,'partition_only':first,'merge_only':merge}
        for f in variants.values():
            for _ in range(10): f()
        gold=dense_reference(q,k,v,table,lens)
        a=api().clone(); n=native()
        torch.testing.assert_close(a.float(),gold,atol=.02,rtol=.02)
        torch.testing.assert_close(bare(),a,atol=0,rtol=0)
        case={'model_geometry':label,'context':length,'plan':asdict(plan),
            'native_equal':torch.equal(a,n),'max_abs_vs_fp32':(a.float()-gold).abs().max().item(),
            'workspace_bytes':r.partials.numel()*4+r.lse.numel()*4,
            'partition_programs':kv*plan.splits,'merge_programs':heads}
        cases.append(case)
        graphs={}
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        for name,f in variants.items():
            for unroll in (1,8,32):
                with torch.cuda.stream(stream):
                    for _ in range(3): f()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(unroll): result=f()
                graphs[name,unroll]=graph
        rng=random.Random(20261103)
        for repeat in range(5):
            order=list(graphs); rng.shuffle(order)
            for name,unroll in order:
                graph=graphs[name,unroll]
                for _ in range(3): graph.replay()
                torch.cuda.synchronize()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter(); begin.record()
                for _ in range(50): graph.replay()
                submitted=time.perf_counter(); end.record(); end.synchronize()
                row={'model_geometry':label,'context':length,'repeat':repeat,'variant':name,'unroll':unroll,
                    'device_interval_us':begin.elapsed_time(end)*1000/(50*unroll),
                    'host_submission_us_per_replay':(submitted-wall)*1e6/50,
                    'synchronized_wall_us_per_operator':(time.perf_counter()-wall)*1e6/(50*unroll)}
                allrows.append(row)
            host={}
            for name,f in [('validate_only',lambda:_validate(q,k,v,table,lens,r,'triton_base',scale)),
                           ('allocate_only',lambda:allocate_matched_resources(spec,'cuda')),
                           ('native',native),('matched_api',api),('matched_bare',bare),
                           ('allocate_and_matched',lambda:matched_decode(q,k,v,table,lens,resources=allocate_matched_resources(spec,'cuda')))]:
                samples=[]
                for _ in range(20):
                    torch.cuda.synchronize(); start=time.perf_counter(); f(); torch.cuda.synchronize()
                    samples.append((time.perf_counter()-start)*1e6)
                host[name]=samples
            write_json(args.output/f'{label}-l{length}-host-r{repeat}.json',host)
            print(label,length,'repeat',repeat,flush=True)
        for name in ('native','matched_api'):
            for unroll in (0,1,32):
                tag=f'{label}-l{length}-{name}-'+('eager' if unroll==0 else f'graph{unroll}')
                f=variants[name] if unroll==0 else graphs[name,unroll].replay
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        with torch.profiler.record_function(tag): f()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.output/(tag+'-trace.json')))
        graphs.clear()
        torch.cuda.synchronize()
    write_json(args.output/'summary.json',{'status':'completed','cases':cases,'rows':allrows,
        'notes':['Stage-only timings are diagnostic and cannot simply be subtracted from complete calls.',
                 'Unrolled Graphs repeat identical metadata and KV; they measure steady-state operator cost, not model generation.',
                 'CUDA-event intervals include execution gaps; profiler traces are separate from formal timing.']})


if __name__=='__main__': main()
