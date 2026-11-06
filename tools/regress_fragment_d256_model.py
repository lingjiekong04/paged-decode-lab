"""Experimental model callback; native, original matched, shared D256 eager/Graph.

Records numerical regression only. Model/session public defaults are unchanged.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
from _common import ROOT,write_json,fingerprint,model_identity
from fragment_d256_common import prepare,complete,strict_equal


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=32);args=p.parse_args()
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from transformers import AttentionInterface
    from paged_decode.hybrid import HybridRunner,_attention
    from paged_decode.matched_splitkv import allocate_matched_resources,matched_decode
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'environment.json',fingerprint())
    write_json(args.output/'arguments.json',{'model_identity':model_identity(args.model),'tokens':args.tokens,
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),ROOT/'tools/fragment_d256_common.py',ROOT/'tools/fragment_asm_d256.py',ROOT/'tools/fragment_partition_asm_d256.py',ROOT/'src/paged_decode/experimental_fragment_partition.py')},
        'gpu_processes_at_start':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),
        'performance_claim':False})
    runner=HybridRunner(args.model,backend='flash_attn',mode='eager',batch=1,max_context=1280)
    assert runner.spec.head_dim==256
    qbuffer=torch.empty((1,runner.spec.num_heads,256),device=runner.device,dtype=torch.bfloat16)
    resource={index:allocate_matched_resources(runner.spec,runner.device) for index in runner.cache.pages}
    for index,(k,v) in runner.cache.pages.items():prepare(qbuffer,k,v,runner.cache.tables,runner.cache.lengths,resource[index],'shared')
    active=['native']
    def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        if runner.cache.is_prefill or active[0]=='native':return _attention(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
        if scaling!=runner.spec.head_dim**-.5:raise ValueError('unsupported model scale')
        qbuffer.copy_(query.squeeze(2));k,v=runner.cache.pages[module.layer_idx];r=resource[module.layer_idx]
        if active[0]=='original':out=matched_decode(qbuffer,k,v,runner.cache.tables,runner.cache.lengths,resources=r,scale=scaling)
        else:out=complete(qbuffer,k,v,runner.cache.tables,runner.cache.lengths,r,'shared')
        return out.unsqueeze(1),None
    name='paged_decode_lab_experimental_d256';AttentionInterface.register(name,attention)
    for layer in runner.text.layers:
        if layer.layer_type=='full_attention':layer.self_attn.config._attn_implementation=name
    prompts=json.loads((ROOT/'configs/prompts.json').read_text(encoding='utf-8'))
    inputs=[runner.tokenizer.apply_chat_template([{'role':'user','content':x}],tokenize=True,add_generation_prompt=True,enable_thinking=False) for x in prompts]
    inputs=[x['input_ids'] if isinstance(x,dict) or hasattr(x,'keys') else x for x in inputs]
    seed=inputs[0]
    inputs += [(seed*((n+len(seed)-1)//len(seed)))[:n] for n in (255,1023)]
    reference={};checks=[];free_checks=[]
    for backend in ('native','original','shared'):
        for mode in ('eager','graph'):
            active[0]=backend;runner.graph=None;runner.mode=mode
            if mode=='graph':runner._capture()
            print('CONFIG',backend,mode,flush=True)
            for prompt_id,ids in enumerate(inputs):
                gold=reference.get(prompt_id)
                forced=None if gold is None else [[x] for x in gold['token_ids'][0]]
                result=runner.generate([ids],args.tokens,forced=forced,observe=True)
                if gold is None:reference[prompt_id]=result;gold=result
                steps=[]
                for step,(a,b) in enumerate(zip(result['logits'],gold['logits'])):
                    same=strict_equal(torch,a,b);delta=a.float()-b.float()
                    steps.append({'step':step,'bytes_equal':same,'max_abs':delta.abs().max().item(),'rmse':delta.square().mean().sqrt().item(),
                        'argmax_equal':torch.equal(a.argmax(-1),b.argmax(-1)),
                        'reference_sha256':hashlib.sha256(b.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                        'actual_sha256':hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
                    if not same:
                        torch.save({'reference':b,'actual':a,'step':step,'input_ids':ids},args.output/f'first-difference-{backend}-{mode}-p{prompt_id}.pt')
                        write_json(args.output/'failure.json',{'backend':backend,'mode':mode,'prompt':prompt_id,'steps':steps})
                        raise RuntimeError('logits byte divergence; first step retained')
                row={'backend':backend,'mode':mode,'prompt':prompt_id,'input_ids':ids,'steps':steps}
                checks.append(row);write_json(args.output/f'logits-{backend}-{mode}-p{prompt_id}.json',row)
                free=runner.generate([ids],args.tokens)
                equal=free['token_ids']==gold['token_ids']
                row={'backend':backend,'mode':mode,'prompt':prompt_id,'tokens_equal':equal,'token_ids':free['token_ids'],'text':free['text']}
                free_checks.append(row);write_json(args.output/f'greedy-{backend}-{mode}-p{prompt_id}.json',row)
                assert equal,'free greedy token divergence'
                print('PROMPT passed',backend,mode,prompt_id,flush=True)
    write_json(args.output/'summary.json',{'status':'passed','model':model_identity(args.model),'prompt_count':len(inputs),
        'logit_steps':sum(len(r['steps']) for r in checks),'free_runs':len(free_checks),'checks':checks,'free_checks':free_checks,
        'head_dim':runner.spec.head_dim,'full_attention_layers':len(resource),'production_defaults_changed':False})


if __name__=='__main__':main()
