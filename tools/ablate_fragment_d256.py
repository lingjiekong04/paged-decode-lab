"""Paged shared-four-state partition experiment; all launches use four warps."""
import argparse,hashlib,json,os,random,subprocess,time,traceback
from pathlib import Path
from dataclasses import asdict
from _common import require_gpu,fingerprint,write_json,ROOT
from fragment_d256_common import partition,merge,complete,prepare,strict_equal
from cubin_text import fingerprint as text_fingerprint


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
    background=pids()
    if background and args.task=='benchmark':raise RuntimeError('selected GPU is busy')
    torch=require_gpu();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.empty(1,device='cuda');torch.cuda.synchronize();own=pids()-background
    if len(own)!=1:raise RuntimeError('cannot establish exclusive host GPU PID')
    def guard():
        if not own.issubset(pids()):raise RuntimeError('own GPU context disappeared')
        return query(['--query-gpu=index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'])
    from paged_decode.spec import DecodeSpec
    from paged_decode.matched_splitkv import allocate_matched_resources,matched_decode
    from paged_decode.torch_reference import dense_reference
    from flash_attn import flash_attn_with_kvcache
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('fragment_d256_common.py'),ROOT/'src/paged_decode/experimental_fragment_partition.py',ROOT/'src/paged_decode/matched_splitkv.py',ROOT/'tools/warp_ablation_common.py',ROOT/'tools/fragment_asm.py',ROOT/'tools/fragment_partition_asm.py')}
    write_json(args.output/'arguments.json',{'task':args.task,'source_sha256':sources,'gpu_uuid':uuid,
        'host_pids':list(own),'background_pids':list(background),'variants':['baseline','shared'],'partition_warps':4,'merge_warps':4,'repeats':5,'replays':20,
        'unrolls':[32,128],'primary_unroll':128,'seed':20261104,
        'fp_fusion':True,'native_splits':'same explicit split count as the experiment plan','strict_byte_comparison':True,
        'controlled':'Only ordered-denominator update changes; identical warps=4, input, tile, split plan and merge launch.'})
    def make(b,n,d,hq=32,hkv=8,lengths=None,num_splits=0):
        torch.manual_seed(123)
        spec=DecodeSpec(b,hq,hkv,d,n)
        q=torch.randn(b,hq,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(b*spec.table_width,256,hkv,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.randperm(b*spec.table_width,device='cuda').int().reshape(b,spec.table_width)
        lens=torch.tensor(lengths if lengths is not None else [n]*b,device='cuda',dtype=torch.int32)
        r={w:allocate_matched_resources(spec,'cuda',num_splits=num_splits) for w in ('baseline','shared')}
        assert r['baseline'].plan==r['shared'].plan
        tensors=(q,k,v,table,lens)
        for w in r:prepare(*tensors,r[w],w)
        return spec,tensors,r
    def score(actual,gold,native=None):
        delta=actual.float()-gold.float();finite=bool(torch.isfinite(actual).all())
        return {'finite':finite,'fp32_close':finite and bool(torch.allclose(actual.float(),gold.float(),atol=.02,rtol=.02)),
            'max_abs_fp32':delta.abs().max().item() if finite else None,
            'rmse_fp32':delta.square().mean().sqrt().item() if finite else None,
            'native_equal':strict_equal(torch,actual,native) if native is not None else None}
    def pair(a,b):
        finite=torch.isfinite(a)&torch.isfinite(b)
        return {'bitwise_equal':strict_equal(torch,a,b),'same_finite_mask':strict_equal(torch,torch.isfinite(a),torch.isfinite(b)),
            'max_abs_finite':(a[finite].float()-b[finite].float()).abs().max().item() if finite.any() else 0.}
    if args.task=='correctness':
        checks=[]
        for d in (256,):
            for hq,hkv in ((8,8),(32,8),(16,1)):
                for num_splits in (0,1):
                    guard();tag=f'd{d}-hq{hq}-hkv{hkv}-splits{num_splits}'
                    print('CORRECTNESS',tag,flush=True)
                    values=[0,1,255,256,257,1023,1024,1025]
                    spec,tensors,r=make(8,2051,d,hq,hkv,values,num_splits=num_splits)
                    q,k,v,table,lens=tensors;original=table.clone()
                    for row,length in enumerate(values):
                        table[row,(length+255)//256:]=-1
                        if length%256:
                            page=int(original[row,length//256].item())
                            k[page,length%256:]=float('nan');v[page,length%256:]=float('nan')
                    graphs={};outputs={}
                    for step,updated in enumerate((values,[257,1025,0,2051,1,256,1024,255],[1,0,1025,0,2051,257,256,1],[0]*8)):
                        if step:
                            table.copy_(original.flip(1) if step%2 else original)
                            lens.copy_(torch.tensor(updated,dtype=torch.int32,device='cuda'))
                            q.normal_();k.normal_();v.normal_()
                        gold=dense_reference(*tensors);outs={}
                        selected=torch.nonzero(lens>0).flatten()
                        native=None
                        if selected.numel():
                            native=flash_attn_with_kvcache(q[selected,None],k,v,block_table=table[selected],cache_seqlens=lens[selected],causal=True,num_splits=r['baseline'].plan.splits).squeeze(1).clone()
                        for w in ('baseline','shared'):
                            if step==0:
                                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                                with torch.cuda.stream(stream):
                                    for _ in range(3):complete(*tensors,r[w],w)
                                torch.cuda.current_stream().wait_stream(stream)
                                graph=torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):out=complete(*tensors,r[w],w)
                                graphs[w]=graph;outputs[w]=out
                            for _ in range(3):graphs[w].replay()
                            torch.cuda.synchronize();actual=outputs[w].clone()
                            eager=complete(*tensors,r[w],w).clone();entry=score(actual,gold)
                            entry.update(case=tag,variant=w,step=step,lengths=updated,
                                graph_eager_equal=strict_equal(torch,actual,eager),
                                native_equal=strict_equal(torch,actual[selected],native) if native is not None else None,
                                native_nonempty_rows=int(selected.numel()),
                                all_empty_zero=bool(torch.count_nonzero(actual)==0) if not any(updated) else None)
                            assert entry['fp32_close'] and entry['graph_eager_equal'] and entry['all_empty_zero'] is not False,entry
                            checks.append(entry);outs[w]=actual
                        comparison={'case':tag,'step':step,'comparison':'baseline_vs_shared',
                            'output':pair(outs['baseline'],outs['shared']),
                            'partials':pair(r['baseline'].partials,r['shared'].partials),'lse':pair(r['baseline'].lse,r['shared'].lse)}
                        assert all(comparison[k]['bitwise_equal'] for k in ('output','partials','lse')),comparison
                        checks.append(comparison)
                        production=matched_decode(*tensors,resources=r['baseline']).clone()
                        assert strict_equal(torch,outs['baseline'],production)
                        checks.append({'case':tag,'step':step,'comparison':'baseline_vs_production','output':pair(outs['baseline'],production)})
                    write_json(args.output/f'{tag}.json',{'checks':[c for c in checks if c['case']==tag]})
                    del graphs,graph,outputs,outs,out,actual,eager,q,k,v,table,lens,tensors,r,gold
                    torch.cuda.synchronize();torch.cuda.empty_cache();guard()
        comparisons=[c for c in checks if c.get('comparison')=='baseline_vs_shared']
        write_json(args.output/'summary.json',{'status':'passed','checks':checks,'compile_failures':[],
            'candidate_bitwise_outputs':sum(c['output']['bitwise_equal'] for c in comparisons),'compared_outputs':len(comparisons)})
        return
    cases=[(b,n,d) for d in (256,) for b in (1,8,32) for n in (512,2048,8192)]
    random.Random(20261104).shuffle(cases);allrows=[];allchecks=[];failures=[]
    for b,n,d in cases:
        guard();print('BENCHMARK',b,n,d,flush=True)
        folder=args.output/f'b{b}-l{n}-d{d}';folder.mkdir()
        spec,tensors,r=make(b,n,d);q,k,v,table,lens=tensors
        gold=dense_reference(*tensors)
        native=flash_attn_with_kvcache(q[:,None],k,v,block_table=table,cache_seqlens=lens,causal=True,num_splits=r['baseline'].plan.splits).squeeze(1).clone()
        compiled={};outs={};active=[]
        for w in ('baseline','shared'):
            try:
                complete(*tensors,r[w],w);torch.cuda.synchronize()
                kernel=partition(*tensors,r[w],w)
                compiled[w]={'registers_per_thread':getattr(kernel,'n_regs',None),'spill_count':getattr(kernel,'n_spills',None),
                    'metadata':str(kernel.metadata),'plan':asdict(r[w].plan),'workspace_bytes':(r[w].partials.numel()+r[w].lse.numel())*4,
                    'dynamic_shared':kernel.metadata.shared,'machine_code':text_fingerprint(kernel.asm['cubin'],kernel.metadata.name)}
                for ext in ('ttir','ttgir','llir','ptx','cubin'):
                    data=kernel.asm[ext];(folder/f'partition-{w}.{ext}').write_bytes(data if isinstance(data,bytes) else data.encode())
                import re
                tool='/usr/local/cuda-12.4/bin/cuobjdump'
                resources_text=subprocess.check_output([tool,'--dump-resource-usage',str(folder/f'partition-{w}.cubin')],text=True)
                sass=subprocess.check_output([tool,'--dump-sass',str(folder/f'partition-{w}.cubin')],text=True)
                (folder/f'partition-{w}.resources.txt').write_text(resources_text)
                (folder/f'partition-{w}.sass.txt').write_text(sass)
                compiled[w].update(static_shared=int(re.search(r'SHARED:(\d+)',resources_text)[1]),
                    stack_bytes=int(re.search(r'STACK:(\d+)',resources_text)[1]),
                    static_ldl=len(re.findall(r'\bLDL\b',sass)),static_stl=len(re.findall(r'\bSTL\b',sass)),
                    ptx_shuffle_sites=sum('shfl.sync.' in line and not line.lstrip().startswith('//') for line in kernel.asm['ptx'].splitlines()))
                merge(r[w]);outs[w]=r[w].out.clone();evaluation=score(outs[w],gold,native)
                allchecks.append({'batch':b,'context':n,'dim':d,'variant':w,**evaluation})
                if not evaluation['fp32_close']:raise RuntimeError('FP32 correctness gate failed; no timing')
                if w=='baseline':torch.testing.assert_close(outs[w],matched_decode(*tensors,resources=r[w]),atol=0,rtol=0)
                active.append(w)
            except Exception:
                failures.append({'batch':b,'context':n,'dim':d,'variant':w,'error':traceback.format_exc()})
                raise
        comparison=({'output':pair(outs['baseline'],outs['shared']),'partials':pair(r['baseline'].partials,r['shared'].partials),
            'lse':pair(r['baseline'].lse,r['shared'].lse)} if 'shared' in active else None)
        assert comparison and all(comparison[k]['bitwise_equal'] for k in ('output','partials','lse')),comparison
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
