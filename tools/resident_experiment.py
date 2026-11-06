"""Numerical/grid/model regression of the explicit two-stage D256 candidate."""
import argparse,hashlib,json,random,sys
from dataclasses import asdict
from pathlib import Path
from _common import ROOT,write_json,fingerprint
from profile_fragment_model_stages import ExclusiveGPU
from fragment_resident_common import prepare,partition,complete,strict_equal,VARIANTS
from partition_residency_common import Driver,resource_record


def sources(output):
    names=['tools/resident_experiment.py','tools/fragment_resident_common.py','tools/fragment_d256_common.py',
           'tools/fragment_asm_d256.py','tools/fragment_partition_asm_d256.py','tools/warp_ablation_common.py',
           'src/paged_decode/experimental_fragment_partition.py','src/paged_decode/matched_splitkv.py',
           'src/paged_decode/hybrid.py','tools/regress_fragment_d256_model.py']
    hashes={}
    for name in names:
        data=(ROOT/name).read_bytes();dest=output/'sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    return hashes


def correctness(torch,args,gpu):
    from paged_decode.spec import DecodeSpec
    from paged_decode.matched_splitkv import allocate_matched_resources
    from paged_decode.torch_reference import dense_reference
    from flash_attn import flash_attn_with_kvcache
    torch.manual_seed(123);checks=[];resources=[]
    # Same GQA/split and replay state matrix as the frozen D256 correctness suite.
    for hq,hkv in ((8,8),(32,8),(16,1)):
        for splits in (0,1):
            gpu.guard();spec=DecodeSpec(8,hq,hkv,256,2051)
            q=torch.randn(8,hq,256,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(8*spec.table_width,256,hkv,256,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
            table=torch.randperm(8*spec.table_width,device='cuda').int().view(8,spec.table_width)
            lengths=torch.tensor([0,1,255,256,257,1023,1024,1025],device='cuda',dtype=torch.int32)
            original=table.clone();tensors=(q,k,v,table,lengths)
            rs={x:allocate_matched_resources(spec,'cuda',num_splits=splits) for x in VARIANTS}
            for variant,r in rs.items():
                c=prepare(*tensors,r,variant)
                resources.append({'hq':hq,'hkv':hkv,'explicit_splits':splits,'variant':variant,
                                  'plan':asdict(r.plan),**resource_record(c,Driver())})
            initial=[0,1,255,256,257,1023,1024,1025]
            for row,n in enumerate(initial):
                table[row,(n+255)//256:]=-1
                if n%256:
                    page=int(original[row,n//256]);k[page,n%256:]=float('nan');v[page,n%256:]=float('nan')
            graphs={};outs={}
            states=[initial,[257,1025,0,2051,1,256,1024,255],[1,0,1025,0,2051,257,256,1],[0]*8]
            for step,state in enumerate(states):
                if step:
                    table.copy_(original.flip(1) if step%2 else original)
                    lengths.copy_(torch.tensor(state,dtype=torch.int32,device='cuda'))
                    q.normal_();k.normal_();v.normal_()
                gold=dense_reference(*tensors);selected=torch.nonzero(lengths>0).flatten()
                native=flash_attn_with_kvcache(q[selected,None],k,v,block_table=table[selected],cache_seqlens=lengths[selected],
                            causal=True,num_splits=rs['original'].plan.splits).squeeze(1).clone() if selected.numel() else None
                values={}
                for variant,r in rs.items():
                    if step==0:
                        for _ in range(3):complete(*tensors,r,variant)
                        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):out=complete(*tensors,r,variant)
                        graphs[variant]=graph;outs[variant]=out
                    for _ in range(3):graphs[variant].replay()
                    torch.cuda.synchronize();actual=outs[variant].clone()
                    eager=complete(*tensors,r,variant).clone()
                    assert torch.isfinite(actual).all() and torch.allclose(actual.float(),gold,atol=.02,rtol=.02)
                    assert strict_equal(torch,actual,eager)
                    native_equal=native is None or strict_equal(torch,actual[selected],native)
                    assert native_equal
                    if not any(state):assert torch.count_nonzero(actual)==0
                    checks.append({'hq':hq,'hkv':hkv,'explicit_splits':splits,'step':step,'variant':variant,
                        'lengths':state,'graph_eager_equal':True,'native_equal':True,'finite':True,
                        'max_abs_fp32':(actual.float()-gold).abs().max().item(),
                        'rmse_fp32':(actual.float()-gold).square().mean().sqrt().item()})
                    values[variant]=(actual,r.partials.clone(),r.lse.clone())
                for variant in VARIANTS:
                    assert all(strict_equal(torch,a,b) for a,b in zip(values[variant],values['original']))
                gpu.guard()
            write_json(args.output/f'case-hq{hq}-hkv{hkv}-splits{splits}.json',
                       {'checks':checks[-12:],'all_output_partials_lse_bytes_equal':True})
            print('CORRECT',hq,hkv,splits,flush=True)
            del graphs,outs,rs,values,q,k,v,table,lengths,tensors,gold
            torch.cuda.synchronize();torch.cuda.empty_cache()
    write_json(args.output/'summary.json',{'status':'passed','checks':checks,'resources':resources,
                                          'all_output_partials_lse_bytes_equal':True})


def benchmark(torch,args,gpu):
    from paged_decode.matched_splitkv import allocate_matched_resources
    from diagnose_partition_residency import loaded
    from fragment_resident_common import merge
    rng=random.Random(20261105);rows=[];records=[]
    for n in (512,2048):
        spec,tensors,native=loaded(torch,ROOT/f'results/residency-snapshots-v2/n{n}-layer3.pt')
        rs={x:allocate_matched_resources(spec,'cuda',num_splits=36) for x in VARIANTS}
        kernels={x:prepare(*tensors,r,x) for x,r in rs.items()}
        full_ref=None
        for variant,r in rs.items():
            out=complete(*tensors,r,variant).clone()
            assert strict_equal(torch,out,native)
            values=(r.partials.clone(),r.lse.clone())
            if full_ref is None:full_ref=values
            assert all(strict_equal(torch,a,b) for a,b in zip(values,full_ref))
            records.append({'input_tokens':n,'variant':variant,**resource_record(kernels[variant],Driver())})
        def launch(variant,stage,parts):
            r=rs[variant]
            # Fixed compiled specialization, only launch grid changes.
            if parts==36:return partition(*tensors,r,variant) if stage=='partition' else complete(*tensors,r,variant)
            return grid_launch(*tensors,r,variant,parts)
        graphs={}
        for variant,r in rs.items():
            for stage,parts in [('partition',g) for g in (18,24,27,28,32,33,36)]+[('full',36)]:
                if parts<36:
                    r.partials.fill_(float('nan'));r.lse.fill_(float('nan'))
                    grid_launch(*tensors,r,variant,parts)
                    assert strict_equal(torch,r.partials[:,:,:parts],full_ref[0][:,:,:parts])
                    assert strict_equal(torch,r.lse[:,:,:parts],full_ref[1][:,:,:parts])
                    assert torch.isnan(r.partials[:,:,parts:]).all() and torch.isnan(r.lse[:,:,parts:]).all()
                for unroll in (32,128):
                    for _ in range(3):launch(variant,stage,parts)
                    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(unroll):launch(variant,stage,parts)
                    graphs[variant,stage,parts,unroll]=graph
        for repeat in range(5):
            before=gpu.guard();order=list(graphs);rng.shuffle(order);samples=[]
            for ordinal,(variant,stage,parts,unroll) in enumerate(order):
                graph=graphs[variant,stage,parts,unroll]
                for _ in range(3):graph.replay()
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):graph.replay()
                end.record();end.synchronize()
                samples.append(dict(input_tokens=n,context=n+16,variant=variant,stage=stage,parts=parts,unroll=unroll,
                                    repeat=repeat,order=ordinal,us_per_call=start.elapsed_time(end)*1000/(unroll*20)))
            after=gpu.guard();rows.extend(samples)
            write_json(args.output/f'round-n{n}-r{repeat}.json',{'before':before,'after':after,'rows':samples})
            print('TIMED',n,repeat,flush=True)
        del graphs,rs,kernels
    write_json(args.output/'summary.json',{'status':'passed','rows':rows,'resources':records,'real_tensor_native_bytes_equal':True})


def grid_launch(q,k,v,table,lengths,r,variant,parts):
    # Grid is not an arithmetic/compile option: retain the existing prepared
    # JIT specialization and audit it using the same launch argument builder.
    from fragment_resident_common import launch_grid
    return launch_grid(q,k,v,table,lengths,r,variant,parts)


def model(torch,args,gpu):
    import regress_fragment_d256_model as frozen
    mapping={'baseline':'original','shared':'shared_s2'};audits=[]
    def adapt_prepare(q,k,v,table,lengths,r,variant,**kw):
        gpu.guard();c=prepare(q,k,v,table,lengths,r,mapping[variant]);audits.append(resource_record(c,Driver()));return c
    frozen.prepare=adapt_prepare
    frozen.complete=lambda q,k,v,t,l,r,variant:complete(q,k,v,t,l,r,mapping[variant])
    frozen.write_json=write_json
    sys.argv=['regress_fragment_d256_model.py','--model',args.model,'--output',str(args.output),'--tokens','32']
    frozen.main();gpu.guard()
    write_json(args.output/'resident-experiment.json',{'legacy_shared_label':'shared_s2','resources':audits,
        'source_sha256':sources(args.output),'scope':'Frozen real prompts, boundary prompts, eager/Graph, forced logits and free greedy suite.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=('correctness','benchmark','model'),required=True)
    p.add_argument('--model');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    # Frozen model runner creates its own output folder; guard does not create it.
    if args.task!='model':args.output.mkdir(parents=True,exist_ok=False)
    gpu=ExclusiveGPU(args.output)
    try:
        import torch
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        if args.task!='model':
            write_json(args.output/'environment.json',fingerprint())
            write_json(args.output/'arguments.json',{'task':args.task,'gpu_uuid':gpu.uuid,'host_pids':sorted(gpu.own),
                       'source_sha256':sources(args.output),'variants':VARIANTS,'scope':'Only pipeline num_stages 3→2 changes.'})
        with torch.inference_mode():
            if args.task=='correctness':correctness(torch,args,gpu)
            elif args.task=='benchmark':benchmark(torch,args,gpu)
            else:model(torch,args,gpu)
    finally:gpu.close()


if __name__=='__main__':main()
