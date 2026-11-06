"""Same fixed-budget model protocol, adding explicit shared_s3/shared_s2 ablation."""
import argparse,gc,hashlib,json,random,statistics,time
from pathlib import Path
from _common import ROOT,write_json,fingerprint,model_identity
from profile_fragment_model_stages import ExclusiveGPU
from fragment_resident_common import prepare,complete,strict_equal
from partition_residency_common import Driver,resource_record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);gpu=ExclusiveGPU(args.output)
    try:
        import torch
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode():run(torch,args,gpu)
    finally:gpu.close()


def run(torch,args,gpu):
    from transformers import AttentionInterface
    from paged_decode.hybrid import HybridRunner,_attention
    from paged_decode.matched_splitkv import allocate_matched_resources,matched_decode
    write_json(args.output/'environment.json',fingerprint())
    source_names=['tools/benchmark_fragment_resident_models.py','tools/fragment_resident_common.py',
        'tools/fragment_d256_common.py','tools/fragment_partition_asm_d256.py','tools/fragment_asm_d256.py',
        'tools/warp_ablation_common.py','src/paged_decode/experimental_fragment_partition.py',
        'src/paged_decode/matched_splitkv.py','src/paged_decode/hybrid.py','tools/profile_fragment_model_stages.py']
    hashes={}
    for name in source_names:
        data=(ROOT/name).read_bytes();dest=args.output/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    identity=model_identity(args.model)
    write_json(args.output/'arguments.json',{'model':identity,'gpu_uuid':gpu.uuid,'host_pids':sorted(gpu.own),
        'source_sha256':hashes,'input_lengths':[512,2048],'output_tokens':64,'repeats':5,'batch':1,'max_context':2112,
        'scope':'Primary wall call has no per-step events; separate event call. Three per-variant workspaces retained in every configuration. '
                'Compilation/capture/warmup excluded; CPU/host shared across the two selected exclusive GPUs.'})
    runner=HybridRunner(args.model,backend='flash_attn',mode='eager',batch=1,max_context=2112)
    qbuffer=torch.empty((1,runner.spec.num_heads,256),device=runner.device,dtype=torch.bfloat16)
    variants=('native','original','shared_s3','shared_s2')
    resources={variant:{i:allocate_matched_resources(runner.spec,runner.device) for i in runner.cache.pages}
               for variant in variants if variant!='native'}
    audits={}
    for variant,by_layer in resources.items():
        for i,(k,v) in runner.cache.pages.items():
            compiled=prepare(qbuffer,k,v,runner.cache.tables,runner.cache.lengths,by_layer[i],variant)
            if variant not in audits:
                audits[variant]=resource_record(compiled,Driver())
                for ext in ('ttgir','ptx','cubin'):
                    data=compiled.asm[ext];(args.output/(variant+'.'+ext)).write_bytes(data if isinstance(data,bytes) else data.encode())
    write_json(args.output/'resources.json',audits)
    active=['native'];calls={x:0 for x in variants}
    def attention(module,query,key,value,mask,scaling,dropout=0.,**kwargs):
        if runner.cache.is_prefill:return _attention(module,query,key,value,mask,scaling,dropout,**kwargs)
        b=active[0];calls[b]+=1
        if b=='native':return _attention(module,query,key,value,mask,scaling,dropout,**kwargs)
        if scaling!=256**-.5:raise ValueError('unsupported scale')
        qbuffer.copy_(query.squeeze(2));k,v=runner.cache.pages[module.layer_idx];r=resources[b][module.layer_idx]
        out=matched_decode(qbuffer,k,v,runner.cache.tables,runner.cache.lengths,resources=r,scale=scaling) if b=='original' else \
            complete(qbuffer,k,v,runner.cache.tables,runner.cache.lengths,r,b)
        return out.unsqueeze(1),None
    name='paged_decode_lab_resident_pipeline_model';AttentionInterface.register(name,attention)
    for layer in runner.text.layers:
        if layer.layer_type=='full_attention':layer.self_attn.config._attn_implementation=name
    def configure(backend,mode):
        active[0]=backend;runner.graph=None;runner.mode=mode
        for attr in ('graph_logits','graph_tokens','graph_position'):
            if hasattr(runner,attr):delattr(runner,attr)
        gc.collect();torch.cuda.empty_cache()
        if mode=='graph':runner._capture()
    rng=random.Random(20261105)
    inputs={n:[rng.choices(range(100,30000),k=n)] for n in (512,2048)}
    write_json(args.output/'inputs.json',inputs)
    references={};validation=[];configs=[(v,m) for v in variants for m in ('eager','graph')]
    for n in inputs:
        configure('native','eager');references[n]=runner.generate(inputs[n],64,observe=True)
    for backend,mode in configs:
        gpu.guard();configure(backend,mode)
        for n in inputs:
            gold=references[n];forced=[[x] for x in gold['token_ids'][0]]
            actual=runner.generate(inputs[n],64,forced=forced,observe=True);checks=[]
            assert len(actual['logits'])==64
            for step,(a,b) in enumerate(zip(actual['logits'],gold['logits'])):
                same=strict_equal(torch,a,b)
                if not same:
                    torch.save({'step':step,'actual':a,'reference':b,'input_ids':inputs[n]},
                               args.output/f'divergence-{backend}-{mode}-n{n}.pt')
                    raise RuntimeError('full logits byte gate failed')
                checks.append({'step':step,'bytes_equal':True,'max_abs':(a.float()-b.float()).abs().max().item()})
            free=runner.generate(inputs[n],64);assert free['token_ids']==gold['token_ids']
            validation.append({'backend':backend,'mode':mode,'input_tokens':n,'steps':checks,'free_tokens_equal':True})
            print('VALIDATED',backend,mode,n,flush=True)
    write_json(args.output/'validation.json',{'rows':validation,'calls':dict(calls)})
    assert all(calls[v]>0 for v in variants)
    rows=[]
    for repeat in range(5):
        order=[(b,m,n) for b,m in configs for n in inputs];rng.shuffle(order)
        samples=[]
        for ordinal,(backend,mode,n) in enumerate(order):
            before=gpu.guard();configure(backend,mode)
            warm=runner.generate(inputs[n],64);assert warm['token_ids']==references[n]['token_ids']
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();actual=runner.generate(inputs[n],64);torch.cuda.synchronize()
            wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
            assert actual['token_ids']==references[n]['token_ids'];between=gpu.guard()
            measured=runner.generate(inputs[n],64,timing=True)
            assert measured['token_ids']==references[n]['token_ids'];after=gpu.guard()
            samples.append({'repeat':repeat,'order':ordinal,'backend':backend,'mode':mode,'input_tokens':n,
                'wall_seconds':wall,'output_tokens':64,'output_tokens_per_second':64/wall,
                'decode_event_ms':measured['decode_ms'],'peak_allocated_bytes':peak,'tokens_equal_native':True,
                'before':before,'between':between,'after':after})
            print('TIMED',repeat,backend,mode,n,wall,flush=True)
        rows.extend(samples);write_json(args.output/f'round{repeat}.json',{'rows':samples})
    write_json(args.output/'summary.json',{'status':'passed','model':identity,'rows':rows,'validation':validation,
        'resources':audits,'callback_invocations':calls,'production_changed':False,'retained_workspaces_shared_across_configs':True})


if __name__=='__main__':main()
