"""Six configurations, fixed KV budget, paired validation and five free repeats."""
import argparse,gc,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
from _common import require_gpu,fingerprint,model_identity,write_json,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--input-tokens',type=int,default=2048)
    p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if min(args.batch,args.input_tokens,args.tokens,args.repeats)<1: p.error('positive sizes required')
    torch=require_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    from transformers import AttentionInterface
    from paged_decode.hybrid import HybridRunner,_attention
    from paged_decode.runtime import allocate_resources
    from paged_decode.matched_splitkv import allocate_matched_resources,matched_decode
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{**vars(args),'output':str(args.output),
        'model_identity':model_identity(args.model),'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch_threads':4,'scope':'fixed batch; official Torch linear attention fallback; text only'})
    runner=HybridRunner(args.model,batch=args.batch,max_context=max(1024,args.input_tokens+args.tokens))
    matched=None
    def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        if runner.cache.is_prefill:
            return _attention(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
        k,v=runner.cache.pages[module.layer_idx]
        out=matched_decode(query.squeeze(2),k,v,runner.cache.tables,runner.cache.lengths,resources=matched,scale=scaling)
        return out.unsqueeze(1),None
    AttentionInterface.register('paged_decode_lab_matched_benchmark',attention)
    def configure(backend,mode):
        nonlocal matched
        runner.graph=None
        for attr in ('graph_logits','graph_tokens','graph_position'):
            if hasattr(runner,attr): delattr(runner,attr)
        runner.resources=None; matched=None
        gc.collect(); torch.cuda.empty_cache()
        runner.backend='flash_attn' if backend=='matched_splitkv' else backend
        runner.mode=mode
        if backend=='matched_splitkv': matched=allocate_matched_resources(runner.spec,runner.device)
        elif backend!='flash_attn': runner.resources=allocate_resources(runner.spec,runner.device,backend)
        for layer in runner.text.layers:
            if layer.layer_type=='full_attention':
                layer.self_attn.config._attn_implementation='paged_decode_lab_matched_benchmark' if matched is not None else 'paged_decode_lab_qwen35'
        if mode=='graph': runner._capture()
    gen=torch.Generator(device='cpu').manual_seed(927)
    prompts=torch.randint(100,30000,(args.batch,args.input_tokens),generator=gen).tolist()
    prompt_sha=hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    baseline=None; native_tokens=None; rows=[]; validation=[]
    for backend in ('flash_attn','triton_tc_splitk','matched_splitkv'):
        for mode in ('eager','graph'):
            print('CONFIG',backend,mode,flush=True)
            configure(backend,mode)
            forced=None if native_tokens is None else [[r[s] for r in native_tokens] for s in range(args.tokens)]
            check=runner.generate(prompts,args.tokens,forced=forced,observe=True)
            if baseline is None: baseline=check['logits']; native_tokens=check['token_ids']
            steps=[]
            for step,(a,b) in enumerate(zip(check['logits'],baseline)):
                diff=a.float()-b.float()
                steps.append({'step':step,'bitwise_equal':torch.equal(a,b),'max_abs':diff.abs().max().item(),
                    'rmse':diff.square().mean().sqrt().item(),'argmax_equal':torch.equal(a.argmax(-1),b.argmax(-1))})
            val={'backend':backend,'mode':mode,'steps':steps,'all_logits_equal':all(s['bitwise_equal'] for s in steps)}
            validation.append(val)
            write_json(args.output/f'validation-{backend}-{mode}.json',val)
            if backend=='matched_splitkv' and not val['all_logits_equal']: raise RuntimeError('matched regression failed; benchmark stopped')
            # Compilation, capture, CPU logits copies and warmup are outside formal timing.
            warm=runner.generate(prompts,args.tokens)
            for repeat in range(args.repeats):
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                begin=time.perf_counter()
                result=runner.generate(prompts,args.tokens,timing=True)
                elapsed=time.perf_counter()-begin
                row={'backend':backend,'mode':mode,'repeat':repeat,'batch':args.batch,'input_tokens':args.input_tokens,
                    'output_tokens':args.tokens,'prompt_sha256':prompt_sha,'generation_wall_seconds':elapsed,
                    'output_tokens_per_second':args.batch*args.tokens/elapsed,'decode_model_ms_raw':result['decode_ms'],
                    'decode_model_ms':summarize(result['decode_ms']),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'workspace_bytes':matched.partials.numel()*4+matched.lse.numel()*4 if matched else runner.spec.split_workspace_bytes if backend=='triton_tc_splitk' else 0,
                    'plan':asdict(matched.plan) if matched else None,'kv_table_width':runner.spec.table_width,
                    'free_tokens_match_native':result['token_ids']==native_tokens,'token_ids':result['token_ids']}
                rows.append(row)
                write_json(args.output/f'bench-{backend}-{mode}-r{repeat}.json',row)
            print(backend,mode,'TPS',summarize([r['output_tokens_per_second'] for r in rows[-args.repeats:]]),flush=True)
    write_json(args.output/'summary.json',{'status':'completed','rows':rows,'validation':validation,
        'note':'Free greedy fixed output length; fixed1024 may follow different tokens. Decode intervals include execution gaps; full wall includes prefill and token selection.'})


if __name__=='__main__': main()
