"""Fast synthetic smoke for launch-reservation control, not model timing."""
import argparse
from pathlib import Path
from _common import write_json
from profile_fragment_model_stages import ExclusiveGPU
from partition_residency_common import Driver,synthetic,build,resource_record,strict_equal


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);gpu=ExclusiveGPU(args.output)
    try:
        import torch
        from warp_ablation_common import merge
        from flash_attn import flash_attn_with_kvcache
        torch.set_num_threads(4)
        with torch.inference_mode():
            d=Driver();spec,tensors=synthetic(torch)
            q,k,v,table,lengths=tensors
            native=flash_attn_with_kvcache(q[:,None],k,v,cache_seqlens=lengths,block_table=table,causal=True).squeeze(1).clone()
            results={};calls={}
            for variant in ('baseline','shared'):
                r,c,fn=build(spec,tensors,variant,d);fn();merge(r)
                assert strict_equal(torch,r.out,native)
                results[variant]=resource_record(c,d);calls[variant]=fn
                if variant=='baseline':
                    partial=r.partials.clone();lse=r.lse.clone();packed=tuple(c.packed_metadata)
                    fn(dynamic=c.metadata.shared+4352);merge(r)
                    assert strict_equal(torch,r.out,native) and strict_equal(torch,r.partials,partial) and strict_equal(torch,r.lse,lse)
                    assert tuple(c.packed_metadata)==packed
                    results['baseline_padded_active_blocks']=d.occupancy(c.function,c.metadata.shared+4352)
                    calls['padded']=lambda fn=fn,c=c:fn(dynamic=c.metadata.shared+4352)
            timing=[]
            for name,fn in calls.items():
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(128):fn()
                for _ in range(3):graph.replay()
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(20):graph.replay()
                b.record();b.synchronize()
                timing.append({'variant':name,'us':a.elapsed_time(b)*1000/2560})
            gpu.guard()
            write_json(args.output/'summary.json',{'status':'passed','resources':results,'timing':timing,
                       'scope':'Synthetic preflight only; formal real-tensor campaign is separate.'})
            print(timing,flush=True)
    finally:gpu.close()


if __name__=='__main__':main()
