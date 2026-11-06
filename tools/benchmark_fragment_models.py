"""Exclusive selected-GPU model benchmark; separate wall and event measurements."""
import argparse,gc,hashlib,json,os,random,statistics,subprocess,threading,time
from pathlib import Path
from _common import ROOT,write_json,fingerprint,model_identity
from fragment_d256_common import prepare,complete,strict_equal


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--repeats',type=int,default=5);args=p.parse_args()
    def query(fields,kind='gpu'):
        return subprocess.check_output(['nvidia-smi',f'--query-{kind}='+fields,'--format=csv,noheader'],text=True,timeout=20)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    uuid=next(u for i,u in [list(map(str.strip,s.split(','))) for s in query('index,uuid').splitlines()] if i==visible or u==visible)
    def pids():return {int(p) for s in query('pid,gpu_uuid','compute-apps').splitlines() for p,u in [list(map(str.strip,s.split(',')))] if u==uuid}
    if pids():raise RuntimeError('selected GPU is not idle; no job will be stopped')
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    allocation=torch.empty(1,device='cuda');torch.cuda.synchronize();own=pids()
    if len(own)!=1:raise RuntimeError('cannot identify own exclusive host GPU PID')
    args.output.mkdir(parents=True,exist_ok=False)
    stop=threading.Event();violations=[];monitor_rows=[]
    def monitor():
        while not stop.wait(1):
            try:
                found=pids();monitor_rows.append({'monotonic':time.monotonic(),'pids':sorted(found)})
                if found!=own:violations.append({'pids':sorted(found)})
            except Exception as e:violations.append({'monitor_error':repr(e)})
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    def guard():
        found=pids()
        if found!=own or violations:
            write_json(args.output/'contamination.json',{'pids':sorted(found),'violations':violations})
            raise RuntimeError('GPU ownership changed; this campaign is invalid')
        return query('index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw')
    try:
        run(args,torch,uuid,own,guard)
    finally:
        stop.set();worker.join(timeout=25)
        write_json(args.output/'gpu-monitor.json',{'rows':monitor_rows,'violations':violations})


def run(args,torch,uuid,own,guard):
    from transformers import AttentionInterface
    from paged_decode.hybrid import HybridRunner,_attention
    from paged_decode.matched_splitkv import allocate_matched_resources,matched_decode
    sources=[Path(__file__),ROOT/'tools/fragment_d256_common.py',ROOT/'tools/fragment_partition_asm_d256.py',ROOT/'tools/fragment_asm_d256.py',ROOT/'src/paged_decode/experimental_fragment_partition.py',ROOT/'src/paged_decode/hybrid.py']
    write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{'model':model_identity(args.model),'gpu_uuid':uuid,'host_pids':sorted(own),
        'input_lengths':[512,2048],'output_tokens':args.tokens,'repeats':args.repeats,'batch':1,'max_context':2048+args.tokens,
        'seed':20261105,'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'scope':'Wall run has no per-step events; separate instrumented run measures decode events. All workspaces retained identically across configs.'})
    runner=HybridRunner(args.model,backend='flash_attn',mode='eager',batch=1,max_context=2048+args.tokens)
    qbuffer=torch.empty((1,runner.spec.num_heads,runner.spec.head_dim),device=runner.device,dtype=torch.bfloat16)
    resource={i:allocate_matched_resources(runner.spec,runner.device) for i in runner.cache.pages}
    for i,(k,v) in runner.cache.pages.items():prepare(qbuffer,k,v,runner.cache.tables,runner.cache.lengths,resource[i],'shared')
    active=['native'];calls={'native':0,'original':0,'shared':0}
    def attention(module,query,key,value,mask,scaling,dropout=0.,**kwargs):
        if runner.cache.is_prefill:return _attention(module,query,key,value,mask,scaling,dropout,**kwargs)
        calls[active[0]]+=1
        if active[0]=='native':return _attention(module,query,key,value,mask,scaling,dropout,**kwargs)
        if scaling!=runner.spec.head_dim**-.5:raise ValueError('unsupported attention scale')
        qbuffer.copy_(query.squeeze(2));k,v=runner.cache.pages[module.layer_idx];r=resource[module.layer_idx]
        out=matched_decode(qbuffer,k,v,runner.cache.tables,runner.cache.lengths,resources=r,scale=scaling) if active[0]=='original' else complete(qbuffer,k,v,runner.cache.tables,runner.cache.lengths,r,'shared')
        return out.unsqueeze(1),None
    name='paged_decode_lab_fragment_model_benchmark';AttentionInterface.register(name,attention)
    for layer in runner.text.layers:
        if layer.layer_type=='full_attention':layer.self_attn.config._attn_implementation=name
    configs=[(backend,mode) for backend in ('native','original','shared') for mode in ('eager','graph')]
    def configure(backend,mode):
        active[0]=backend;runner.graph=None;runner.mode=mode
        for attr in ('graph_logits','graph_tokens','graph_position'):
            if hasattr(runner,attr):delattr(runner,attr)
        gc.collect();torch.cuda.empty_cache()
        if mode=='graph':runner._capture()
    rng=random.Random(20261105)
    inputs={n:[rng.choices(range(100,30000),k=n)] for n in (512,2048)}
    write_json(args.output/'inputs.json',inputs)
    reference={};validation=[]
    for n in inputs:
        configure('native','eager');reference[n]=runner.generate(inputs[n],args.tokens,observe=True)
    for backend,mode in configs:
        guard();configure(backend,mode)
        for n in inputs:
            gold=reference[n];forced=[[t] for t in gold['token_ids'][0]]
            actual=runner.generate(inputs[n],args.tokens,forced=forced,observe=True);steps=[]
            for step,(a,b) in enumerate(zip(actual['logits'],gold['logits'])):
                same=strict_equal(torch,a,b)
                steps.append({'step':step,'bytes_equal':same,'max_abs':(a.float()-b.float()).abs().max().item()})
                if not same:
                    torch.save({'actual':a,'reference':b,'step':step},args.output/f'divergence-{backend}-{mode}-{n}.pt')
                    raise RuntimeError('logits byte gate failed; no timing')
            free=runner.generate(inputs[n],args.tokens)
            assert free['token_ids']==gold['token_ids'],'free greedy gate failed'
            validation.append({'backend':backend,'mode':mode,'input_tokens':n,'steps':steps,'free_tokens_equal':True})
            print('VALIDATED',backend,mode,n,flush=True)
    write_json(args.output/'validation.json',{'rows':validation,'attention_callback_invocations':dict(calls)})
    assert all(calls[b]>0 for b in calls)
    rows=[]
    for repeat in range(args.repeats):
        order=[(b,m,n) for b,m in configs for n in inputs];rng.shuffle(order)
        for ordinal,(backend,mode,n) in enumerate(order):
            before=guard();configure(backend,mode)
            warm=runner.generate(inputs[n],args.tokens);assert warm['token_ids']==reference[n]['token_ids']
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter();result=runner.generate(inputs[n],args.tokens);torch.cuda.synchronize()
            wall=time.perf_counter()-begin;peak=torch.cuda.max_memory_allocated()
            assert result['token_ids']==reference[n]['token_ids']
            between=guard()
            measured=runner.generate(inputs[n],args.tokens,timing=True)
            assert measured['token_ids']==reference[n]['token_ids']
            after=guard()
            row={'backend':backend,'mode':mode,'input_tokens':n,'output_tokens':args.tokens,'batch':1,'repeat':repeat,'order':ordinal,
                'wall_seconds':wall,'output_tokens_per_second':args.tokens/wall,'decode_event_ms':measured['decode_ms'],
                'decode_event_median_ms':statistics.median(measured['decode_ms']),'peak_allocated_bytes':peak,
                'retained_workspace_bytes':sum((r.partials.numel()+r.lse.numel())*4 for r in resource.values()),
                'kv_table_width':runner.spec.table_width,'tokens_equal_native':True,'before':before,'between':between,'after':after}
            rows.append(row);write_json(args.output/f'r{repeat}-{backend}-{mode}-n{n}.json',row)
            print('TIMED',repeat,backend,mode,n,wall,flush=True)
    write_json(args.output/'summary.json',{'status':'passed','rows':rows,'validation':validation,'callback_invocations':calls,
        'model':model_identity(args.model),'shared_workspaces_retained':True,'production_changed':False})


if __name__=='__main__':main()
