"""Isolate local normalization, LSE and merge errors on saved learned Q/K/V."""
import argparse
from pathlib import Path
from _common import require_gpu, write_json, fingerprint
import triton
import triton.language as tl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--location', default='step75/attention31')
    parser.add_argument('--capacity', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch = require_gpu()
    from flash_attn import flash_attn_with_kvcache
    from paged_decode.spec import DecodeSpec
    from paged_decode.matched_splitkv import allocate_matched_resources, matched_decode, _lse_merge
    from dataclasses import asdict
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'environment.json', fingerprint())
    data = torch.load(args.input, map_location='cpu', weights_only=True)['snapshots'][args.location]
    q, keys, vals = (data[key].cuda() for key in ('q','k','v'))
    batch, length, kv, dim = keys.shape
    spec = DecodeSpec(batch,q.shape[1],kv,dim,args.capacity)
    resources = allocate_matched_resources(spec,'cuda')

    def pack(keys, vals):
        n = keys.shape[1]
        width = max(1,(n+255)//256)
        table = torch.arange(batch*width,device='cuda',dtype=torch.int32).view(batch,width)
        k,v = (torch.zeros(batch*width,256,kv,dim,device='cuda',dtype=torch.bfloat16) for _ in range(2))
        pos = torch.arange(n,device='cuda')
        k[table[:,pos//256].long(),pos[None,:]%256] = keys
        v[table[:,pos//256].long(),pos[None,:]%256] = vals
        lens = torch.full((batch,),n,device='cuda',dtype=torch.int32)
        return k,v,table,lens

    k,v,table,lens = pack(torch.nn.functional.pad(keys,(0,0,0,0,0,spec.table_width*256-length)),
                           torch.nn.functional.pad(vals,(0,0,0,0,0,spec.table_width*256-length)))
    lens.fill_(length)
    def native(k,v,table,lens,splits):
        out,lse = flash_attn_with_kvcache(q[:,None],k,v,block_table=table,cache_seqlens=lens,
            causal=True,num_splits=splits,return_softmax_lse=True)
        return out[:,0],lse.reshape(batch,q.shape[1])
    actual = matched_decode(q,k,v,table,lens,resources=resources).clone()
    auto, global_lse = native(k,v,table,lens,0)
    oracle_lse = torch.full_like(resources.lse,-float('inf'))
    rows = []
    def metric(a,b):
        finite = torch.isfinite(a)&torch.isfinite(b)
        diff = torch.where(finite,a.float()-b.float(),0.)
        return {'bitwise_equal':torch.equal(a,b),'unequal':int((a!=b).sum()),
            'max_abs_finite':diff.abs().max().item(),'rmse_finite':diff.square().mean().sqrt().item()}
    for p in range(resources.plan.splits):
        start,end = p*resources.plan.tokens_per_split,min(length,(p+1)*resources.plan.tokens_per_split)
        if start>=end: continue
        pk,pv,pt,pl = pack(keys[:,start:end],vals[:,start:end])
        part, lse = native(pk,pv,pt,pl,1)
        oracle_lse[:,:,p].copy_(lse)
        rows.append({'part':p,'tokens':end-start,
            'local_lse':metric(resources.lse[:,:,p],lse),
            'normalized_partial_bf16':metric(resources.partials[:,:,p].bfloat16(),part)})
    original_lse = resources.lse.clone()
    resources.lse.copy_(oracle_lse)
    _lse_merge[(batch,spec.num_heads)](resources.partials,resources.lse,resources.out,
        HQ=spec.num_heads,D=dim,PARTS=resources.plan.splits,
        PAD_PARTS=triton.next_power_of_2(resources.plan.splits),LANES=resources.plan.lse_lanes,num_warps=4)
    oracle_local = resources.out.clone()

    @triton.jit
    def merge_known(Partial,LSE,Global,Out,HQ:tl.constexpr,D:tl.constexpr,P:tl.constexpr):
        b,h=tl.program_id(0),tl.program_id(1)
        d=tl.arange(0,D)
        base=(b*HQ+h)*P
        logsum=tl.load(Global+b*HQ+h)
        acc=tl.full((D,),0.,tl.float32)
        for p in range(P):
            val=tl.load(Partial+(base+p)*D+d)
            weight=tl.exp(tl.load(LSE+base+p)-logsum)
            acc=tl.fma(weight,val,acc)
        tl.store(Out+(b*HQ+h)*D+d,acc)
    global_lse=global_lse.contiguous()
    merge_known[(batch,spec.num_heads)](resources.partials,oracle_lse,global_lse,resources.out,
        HQ=spec.num_heads,D=dim,P=resources.plan.splits,num_warps=4)
    known_global=resources.out.clone()
    result={'location':args.location,'plan':asdict(resources.plan),'parts':rows,
        'local_lse_total':metric(original_lse,oracle_lse),'original_vs_auto':metric(actual,auto),
        'oracle_local_lse_vs_auto':metric(oracle_local,auto),
        'oracle_local_and_global_lse_vs_auto':metric(known_global,auto)}
    write_json(args.output/'summary.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
