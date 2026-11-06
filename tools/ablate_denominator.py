"""Paired ordered-denominator selection experiment; all launches use four warps."""
import argparse,hashlib,json,os,random,subprocess,time,traceback
from pathlib import Path
from dataclasses import asdict
from _common import require_gpu,fingerprint,write_json,ROOT
from denominator_ablation_common import partition,merge,complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=('benchmark','correctness'),required=True)
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
    if pids():raise RuntimeError('selected GPU is busy')
    torch=require_gpu();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.empty(1,device='cuda');torch.cuda.synchronize();own=pids()
    if len(own)!=1:raise RuntimeError('cannot establish exclusive host GPU PID')
    def guard():
        if pids()!=own:raise RuntimeError('GPU sharing detected; discard this attempt')
        return query(['--query-gpu=index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'])
    from paged_decode.spec import DecodeSpec
    from paged_decode.matched_splitkv import allocate_matched_resources,matched_decode
    from paged_decode.torch_reference import dense_reference
    from flash_attn import flash_attn_with_kvcache
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('denominator_ablation_common.py'),ROOT/'src/paged_decode/experimental_denominator.py',ROOT/'src/paged_decode/matched_splitkv.py',ROOT/'tools/warp_ablation_common.py')}
    write_json(args.output/'arguments.json',{'task':args.task,'source_sha256':sources,'gpu_uuid':uuid,
        'host_pids':list(own),'variants':['baseline','select'],'partition_warps':4,'merge_warps':4,'repeats':5,'replays':20,
        'unrolls':[32,128],'primary_unroll':128,'seed':20261104,
        'controlled':'Only probability block selection changes; identical warps=4, input, tile, split plan and merge launch.'})
    def make(b,n,d,hq=32,hkv=8,lengths=None):
        torch.manual_seed(123)
        spec=DecodeSpec(b,hq,hkv,d,n)
        q=torch.randn(b,hq,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(b*spec.table_width,256,hkv,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.randperm(b*spec.table_width,device='cuda').int().reshape(b,spec.table_width)
        lens=torch.tensor(lengths if lengths is not None else [n]*b,device='cuda',dtype=torch.int32)
        r={w:allocate_matched_resources(spec,'cuda') for w in ('baseline','select')}
        assert r['baseline'].plan==r['select'].plan
        return spec,(q,k,v,table,lens),r
    def score(actual,gold,native=None):
        delta=actual.float()-gold.float();finite=bool(torch.isfinite(actual).all())
        return {'finite':finite,'fp32_close':finite and bool(torch.allclose(actual.float(),gold.float(),atol=.02,rtol=.02)),
            'max_abs_fp32':delta.abs().max().item() if finite else None,
            'rmse_fp32':delta.square().mean().sqrt().item() if finite else None,
            'native_equal':torch.equal(actual,native) if native is not None else None}
    def pair(a,b):
        finite=torch.isfinite(a)&torch.isfinite(b)
        return {'bitwise_equal':torch.equal(a,b),'same_finite_mask':torch.equal(torch.isfinite(a),torch.isfinite(b)),
            'max_abs_finite':(a[finite].float()-b[finite].float()).abs().max().item() if finite.any() else 0.}
    if args.task=='correctness':
        checks=[];failures=[]
        for d in (64,128,256):
            for hq,hkv in ((8,8),(32,8),(16,1)):
                guard();tag=f'd{d}-hq{hq}-hkv{hkv}';print('CORRECTNESS',tag,flush=True)
                values=[0,1,255,256,257,1023,1024,1025]
                spec,tensors,r=make(8,2051,d,hq,hkv,values)
                q,k,v,table,lens=tensors;original=table.clone()
                for b,length in enumerate(values):
                    table[b,(length+255)//256:]=-1
                    if length%256:
                        page=int(original[b,length//256].item());k[page,length%256:]=float('nan');v[page,length%256:]=float('nan')
                active=[];graphs={};outputs={}
                for w in ('baseline','select'):
                    try:
                        complete(*tensors,r[w],w);torch.cuda.synchronize();active.append(w)
                    except Exception:
                        failures.append({'case':tag,'variant':w,'error':traceback.format_exc()})
                        if w=='baseline':raise
                for step,updated in enumerate((values,[257,1025,0,2051,1,256,1024,255],[1,0,1025,0,2051,257,256,1],[0]*8)):
                    if step:
                        table.copy_(original.flip(1) if step%2 else original);lens.copy_(torch.tensor(updated,dtype=torch.int32,device='cuda'))
                        q.normal_();k.normal_();v.normal_()
                    gold=dense_reference(*tensors);outs={}
                    for w in active:
                        if step==0:
                            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                for _ in range(3):complete(*tensors,r[w],w)
                            torch.cuda.current_stream().wait_stream(stream)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):out=complete(*tensors,r[w],w)
                            graphs[w]=graph;outputs[w]=out
                        graphs[w].replay();torch.cuda.synchronize();actual=outputs[w].clone()
                        eager=complete(*tensors,r[w],w).clone();entry=score(actual,gold)
                        entry.update(case=tag,variant=w,step=step,lengths=updated,graph_eager_equal=torch.equal(actual,eager),
                            all_empty_zero=bool(torch.count_nonzero(actual)==0) if not any(updated) else None)
                        checks.append(entry);outs[w]=actual
                    if 'baseline' in outs and 'select' in outs:
                        checks.append({'case':tag,'step':step,'comparison':'baseline_vs_select','output':pair(outs['baseline'],outs['select']),
                            'partials':pair(r['baseline'].partials,r['select'].partials),'lse':pair(r['baseline'].lse,r['select'].lse)})
                    production=matched_decode(*tensors,resources=r['baseline']).clone()
                    torch.testing.assert_close(outs['baseline'],production,atol=0,rtol=0)
                    checks.append({'case':tag,'step':step,'comparison':'baseline_vs_production','output':pair(outs['baseline'],production)})
                write_json(args.output/f'{tag}.json',{'checks':[c for c in checks if c['case']==tag],
                    'compile_failures':[f for f in failures if f['case']==tag]})
                del graphs,graph,outputs,outs,out,actual,eager,q,k,v,table,lens,tensors,r,gold
                torch.cuda.synchronize();torch.cuda.empty_cache()
            guard()
        valid=[c for c in checks if 'fp32_close' in c]
        passed=not failures and all(c['fp32_close'] and c['graph_eager_equal'] and c['all_empty_zero'] is not False for c in valid)
        comparisons=[c for c in checks if c.get('comparison')=='baseline_vs_select']
        write_json(args.output/'summary.json',{'status':'passed' if passed else 'failed','checks':checks,'compile_failures':failures,
            'candidate_bitwise_outputs':sum(c['output']['bitwise_equal'] for c in comparisons),'compared_outputs':len(comparisons)})
        if not passed:raise SystemExit(2)
        return
    cases=[(b,n,d) for d in (64,128,256) for b,n in ((1,512),(1,8192),(32,8192))]
    random.Random(20261104).shuffle(cases);allrows=[];allchecks=[];failures=[]
    for b,n,d in cases:
        guard();print('BENCHMARK',b,n,d,flush=True)
        folder=args.output/f'b{b}-l{n}-d{d}';folder.mkdir()
        spec,tensors,r=make(b,n,d);q,k,v,table,lens=tensors
        gold=dense_reference(*tensors)
        native=flash_attn_with_kvcache(q[:,None],k,v,block_table=table,cache_seqlens=lens,causal=True).squeeze(1).clone()
        compiled={};outs={};active=[]
        for w in ('baseline','select'):
            try:
                complete(*tensors,r[w],w);torch.cuda.synchronize()
                kernel=partition(*tensors,r[w],w)
                compiled[w]={'registers_per_thread':getattr(kernel,'n_regs',None),'spill_count':getattr(kernel,'n_spills',None),
                    'metadata':str(kernel.metadata),'plan':asdict(r[w].plan),'workspace_bytes':(r[w].partials.numel()+r[w].lse.numel())*4}
                (folder/f'partition-{w}.ptx').write_bytes(kernel.asm['ptx'].encode())
                merge(r[w]);outs[w]=r[w].out.clone();evaluation=score(outs[w],gold,native)
                allchecks.append({'batch':b,'context':n,'dim':d,'variant':w,**evaluation})
                if not evaluation['fp32_close']:raise RuntimeError('FP32 correctness gate failed; no timing')
                if w=='baseline':torch.testing.assert_close(outs[w],matched_decode(*tensors,resources=r[w]),atol=0,rtol=0)
                active.append(w)
            except Exception:
                failures.append({'batch':b,'context':n,'dim':d,'variant':w,'error':traceback.format_exc()})
                if w=='baseline':raise
        comparison=({'output':pair(outs['baseline'],outs['select']),'partials':pair(r['baseline'].partials,r['select'].partials),
            'lse':pair(r['baseline'].lse,r['select'].lse)} if 'select' in active else None)
        write_json(folder/'case.json',{'batch':b,'context':n,'dim':d,'compiled':compiled,'comparison':comparison,
            'checks':[c for c in allchecks if (c['batch'],c['context'],c['dim'])==(b,n,d)],
            'failures':[f for f in failures if (f['batch'],f['context'],f['dim'])==(b,n,d)]})
        variants={(w,stage):(lambda w=w,stage=stage:partition(*tensors,r[w],w) if stage=='partition' else complete(*tensors,r[w],w))
            for w in active for stage in ('partition','complete')}
        graphs={};stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for (w,stage),call in variants.items():
            for u in (32,128):
                with torch.cuda.stream(stream):
                    for _ in range(3):call()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(u):captured=call()
                graph.replay();torch.cuda.synchronize()
                if stage=='complete':torch.testing.assert_close(captured.float(),gold,atol=.02,rtol=.02)
                graphs[w,stage,u]=graph
        samples=[];snapshots=[];rng=random.Random(20261104)
        for repeat in range(5):
            before=guard();order=list(graphs);rng.shuffle(order)
            for index,(w,stage,u) in enumerate(order):
                graph=graphs[w,stage,u]
                for _ in range(3):graph.replay()
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();start.record()
                for _ in range(20):graph.replay()
                submitted=time.perf_counter();end.record();end.synchronize()
                samples.append({'variant':w,'stage':stage,'unroll':u,'repeat':repeat,'order':index,
                    'device_interval_us':start.elapsed_time(end)*1000/(20*u),'host_submit_us_per_replay':(submitted-wall)*1e6/20})
            snapshots.append({'repeat':repeat,'before':before,'after':guard()})
        write_json(folder/'timing.json',{'rows':samples,'gpu_snapshots':snapshots})
        allrows.extend([{**s,'batch':b,'context':n,'dim':d} for s in samples])
        del graphs,graph,captured,outs,native,q,k,v,table,lens,tensors,r,gold
        torch.cuda.synchronize();torch.cuda.empty_cache()
    write_json(args.output/'summary.json',{'status':'completed_with_rejections' if failures else 'completed',
        'rows':allrows,'checks':allchecks,'rejections':failures,'note':'No production change; compare paired measurements, not different campaigns.'})


if __name__=='__main__':main()
