"""Diagnose the fixed 9B D256 partition's resource/CTA residency boundary."""
import argparse
import ctypes
from pathlib import Path
import hashlib
import json
import random
import statistics
import subprocess

from _common import ROOT,write_json,fingerprint,model_identity
from profile_fragment_model_stages import ExclusiveGPU
from partition_residency_common import Driver,synthetic,build,resource_record,strict_equal


def resources(torch,output):
    driver=Driver()
    spec,tensors=synthetic(torch)
    rows={}
    for variant in ('baseline','shared'):
        r,compiled,launch=build(spec,tensors,variant,driver)
        launch();torch.cuda.synchronize()
        rows[variant]=resource_record(compiled,driver)
        for ext in ('ttgir','llir','ptx','cubin'):
            data=compiled.asm[ext]
            (output/f'{variant}.{ext}').write_bytes(data if isinstance(data,bytes) else data.encode())
        for flag,name in (('--dump-resource-usage','resources'),('--dump-sass','sass')):
            result=subprocess.run(['/usr/local/cuda/bin/cuobjdump',flag,str(output/f'{variant}.cubin')],
                                  capture_output=True,text=True,timeout=30)
            if result.returncode:raise RuntimeError(result.stderr)
            (output/f'{variant}.{name}.txt').write_text(result.stdout)
    write_json(output/'resources.json',{'device':driver.device_resources(),'variants':rows})
    print(json.dumps(rows),flush=True)


def snapshot(torch,args):
    from paged_decode.hybrid import HybridRunner
    runner=HybridRunner(args.model,backend='flash_attn',mode='eager',batch=1,max_context=2112)
    inputs=json.loads((ROOT/'results/stage-large-v1/inputs.json').read_text())
    manifest=[]
    for n in (512,2048):
        seen=[0];saved=[]
        def observe(module,q,k,v,out):
            if module.layer_idx!=3:return
            seen[0]+=1
            if seen[0]!=16:return
            table=runner.cache.tables.cpu().clone()
            lengths=runner.cache.lengths.cpu().clone()
            # Padding is not part of logical KV. Zero it for portable snapshots.
            kc,vc=torch.zeros_like(k,device='cpu'),torch.zeros_like(v,device='cpu')
            source_k,source_v=k.cpu(),v.cpu()
            for logical in range(n+16):
                page=int(table[0,logical//256]);slot=logical%256
                kc[page,slot].copy_(source_k[page,slot]);vc[page,slot].copy_(source_v[page,slot])
            saved.append({'q':q.cpu().contiguous(),'k':kc,'v':vc,'table':table,'lengths':lengths,
                          'native_output':out.cpu().contiguous()})
        runner.attention_observer=observe
        result=runner.generate(inputs[str(n)],17,observe=True)
        assert len(saved)==1 and seen[0]==16
        path=args.output/f'n{n}-layer3.pt'
        torch.save(saved[0],path)
        manifest.append({'input_tokens':n,'context':n+16,'layer':3,'file':path.name,
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'tokens':result['token_ids']})
    write_json(args.output/'snapshots.json',{'model':model_identity(args.model),'rows':manifest,
               'scope':'Native layer 3 at decode step 16; unreferenced KV slots zeroed; fixed capacity 2112.'})


def loaded(torch,path):
    from paged_decode.spec import DecodeSpec
    s=torch.load(path,map_location='cuda',weights_only=True)
    return DecodeSpec(1,16,4,256,2112),tuple(s[k] for k in ('q','k','v','table','lengths')),s['native_output']


def benchmark(torch,args,gpu):
    from warp_ablation_common import merge
    driver=Driver()
    rng=random.Random(20261105)
    allrows=[];checks=[]
    for input_length in (512,2048):
        spec,tensors,native=loaded(torch,args.snapshots/f'n{input_length}-layer3.pt')
        variants={v:build(spec,tensors,v,driver) for v in ('baseline','shared')}
        rec={v:resource_record(c,driver) for v,(r,c,fn) in variants.items()}
        pad=rec['shared']['static_shared_bytes']-rec['baseline']['static_shared_bytes']
        assert pad>0
        baseline_dynamic=rec['baseline']['dynamic_shared_bytes']
        padded_dynamic=baseline_dynamic+pad
        driver.allow_dynamic(variants['baseline'][1].function,padded_dynamic)
        rec['baseline_padded']={**rec['baseline'],'dynamic_shared_bytes':padded_dynamic,
             'driver_max_active_blocks_per_sm':driver.occupancy(variants['baseline'][1].function,padded_dynamic)}
        outputs={}
        for v,(r,c,fn) in variants.items():
            fn();merge(r);outputs[v]=r.out.clone()
            assert strict_equal(torch,outputs[v],native),'native snapshot byte gate failed'
        r,c,fn=variants['baseline']
        partial_ref=r.partials.clone();lse_ref=r.lse.clone()
        fn(dynamic=padded_dynamic);merge(r)
        assert strict_equal(torch,r.out,native)
        assert strict_equal(torch,r.partials,partial_ref) and strict_equal(torch,r.lse,lse_ref)
        checks.append({'input_tokens':input_length,'native_bytes_equal':True,'padded_partials_bytes_equal':True,
                       'resources':rec})
        folder=args.output/f'n{input_length}';folder.mkdir()
        write_json(folder/'resources.json',rec)
        graphs={}
        # Reduced grids are partition-only diagnostics. They deliberately leave
        # unlaunched partials untouched, and do not produce complete attention.
        for parts in (18,24,27,28,32,33,36):
            for label in ('baseline','baseline_padded','shared'):
                key='baseline' if label=='baseline_padded' else label
                r,c,fn=variants[key]
                dynamic=padded_dynamic if label=='baseline_padded' else None
                r.partials.fill_(float('nan'));r.lse.fill_(float('nan'))
                fn(parts,dynamic);torch.cuda.synchronize()
                assert strict_equal(torch,r.partials[:,:,:parts],partial_ref[:,:,:parts])
                assert strict_equal(torch,r.lse[:,:,:parts],lse_ref[:,:,:parts])
                if parts<36:
                    assert torch.isnan(r.partials[:,:,parts:]).all() and torch.isnan(r.lse[:,:,parts:]).all()
                for unroll in (32,128):
                    for _ in range(3):fn(parts,dynamic)
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(unroll):fn(parts,dynamic)
                    graphs[label,parts,unroll]=graph
        rows=[]
        for repeat in range(5):
            before=gpu.guard()
            order=list(graphs);rng.shuffle(order)
            for ordinal,(label,parts,unroll) in enumerate(order):
                graph=graphs[label,parts,unroll]
                for _ in range(3):graph.replay()
                torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):graph.replay()
                end.record();end.synchronize()
                row={'input_tokens':input_length,'context':input_length+16,'variant':label,
                     'grid_parts':parts,'total_ctas':4*parts,'active_ctas':4*min(parts,(input_length+79)//64),
                     'unroll':unroll,'repeat':repeat,'order':ordinal,
                     'us_per_call':start.elapsed_time(end)*1000/(unroll*20)}
                rows.append(row)
            after=gpu.guard()
            write_json(folder/f'round{repeat}.json',{'before':before,'after':after,'rows':rows[-len(order):]})
            print('BENCH',input_length,repeat,flush=True)
        allrows.extend(rows)
        del graphs
    write_json(args.output/'summary.json',{'status':'passed','rows':allrows,'checks':checks,
        'scope':'Fixed compiled instruction streams, grid sweep and unused shared reservation padding. '
                'Reduced grids do not implement a full attention call. Not a model throughput experiment.'})


def profile(torch,args):
    # Prepare and warm up without profiling. The profiler sees one selected
    # partition only; no model loading or compilation is included in counters.
    driver=Driver()
    spec,tensors=synthetic(torch,args.context)
    variant='shared' if args.variant=='shared' else 'baseline'
    r,c,fn=build(spec,tensors,variant,driver)
    dynamic=c.metadata.shared+4352 if args.variant=='baseline_padded' else None
    for _ in range(3):fn(args.parts,dynamic)
    torch.cuda.synchronize()
    libraries={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line}
    if len(libraries)!=1:raise RuntimeError('ambiguous CUDA runtime')
    runtime=ctypes.CDLL(libraries.pop())
    Driver.check(runtime.cudaProfilerStart())
    fn(args.parts,dynamic)
    torch.cuda.synchronize()
    Driver.check(runtime.cudaProfilerStop())
    write_json(args.output/'profile-target.json',{'variant':args.variant,'context':args.context,
               'parts':args.parts,'resources':resource_record(c,driver),'synthetic_values':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('resources','snapshot','benchmark','profile'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model')
    p.add_argument('--snapshots',type=Path)
    p.add_argument('--variant',choices=('baseline','baseline_padded','shared'),default='baseline')
    p.add_argument('--context',type=int,default=2064)
    p.add_argument('--parts',type=int,default=36)
    args=p.parse_args()
    if not 1<=args.parts<=36 or args.context not in (528,2064):p.error('unsupported fixed geometry')
    args.output.mkdir(parents=True,exist_ok=False)
    gpu=ExclusiveGPU(args.output)
    try:
        import torch
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        write_json(args.output/'environment.json',fingerprint())
        sources=[Path(__file__),ROOT/'tools/partition_residency_common.py',ROOT/'tools/fragment_d256_common.py',
                 ROOT/'tools/fragment_asm_d256.py',ROOT/'tools/fragment_partition_asm_d256.py',
                 ROOT/'tools/warp_ablation_common.py',ROOT/'src/paged_decode/matched_splitkv.py',
                 ROOT/'src/paged_decode/experimental_fragment_partition.py',ROOT/'src/paged_decode/hybrid.py',
                 ROOT/'tools/profile_fragment_model_stages.py']
        hashes={}
        for source in sources:
            n=source.relative_to(ROOT)
            dest=args.output/'sources'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
            hashes[n.as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(args.output/'arguments.json',{'task':args.task,'gpu_uuid':gpu.uuid,'host_pids':sorted(gpu.own),
                   'source_sha256':hashes,'variant':args.variant,'context':args.context,'parts':args.parts})
        with torch.inference_mode():
            if args.task=='resources':resources(torch,args.output)
            elif args.task=='snapshot':snapshot(torch,args)
            elif args.task=='benchmark':benchmark(torch,args,gpu)
            else:profile(torch,args)
        gpu.guard()
    finally:
        gpu.close()


if __name__=='__main__':main()
