"""Summarize stage intervals, compiled resources and separate CUDA timelines."""
import argparse,json,statistics as st
from pathlib import Path
from _common import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/smalld-stage-v1')
    p.add_argument('--output',type=Path,default=ROOT/'docs/small-d')
    args=p.parse_args()
    raw=json.loads((args.input/'summary.json').read_text())
    assert raw['status']=='completed' and len(raw['cases'])==9 and len(raw['rows'])==450
    rows=[];traces=[];hosts=[]
    for c in sorted(raw['cases'],key=lambda c:(c['dim'],c['batch'],c['context'])):
        b,n,d=c['batch'],c['context'],c['dim'];folder=args.input/f'b{b}-l{n}-d{d}'
        c['partition_ptx_bytes']=(folder/'partition.ptx').stat().st_size
        timing=json.loads((folder/'timing.json').read_text())
        for v in ('native','matched_api','matched_bare','partition_only','merge_only'):
            values={u:[r['device_interval_us'] for r in timing['rows'] if (r['variant'],r['unroll'])==(v,u)] for u in (32,128)}
            assert all(len(x)==5 for x in values.values())
            primary=st.median(values[128]);delta=abs(st.median(values[32])-primary)/primary
            rows.append({'batch':b,'context':n,'dim':d,'variant':v,'primary_us':primary,
                'range_us':[min(values[128]),max(values[128])],'unroll32_us':st.median(values[32]),
                'convergence_delta':delta,'converged_5pct':delta<=.05})
        for v in ('native','matched_preallocated','bare_preallocated','matched_allocate'):
            selected=[r for r in timing['host_rows'] if r['variant']==v]
            assert len(selected)==5 and all(len(r['synchronized_wall_us'])==20 for r in selected)
            medians=[st.median(r['synchronized_wall_us']) for r in selected]
            hosts.append({'batch':b,'context':n,'dim':d,'variant':v,'median_of_repeat_medians_us':st.median(medians),
                'range_repeat_medians_us':[min(medians),max(medians)]})
        for variant in ('native','matched_api'):
            events=json.loads((folder/f'{variant}-trace.json').read_text())['traceEvents']
            kernels=[e for e in events if e.get('cat')=='kernel' and
                (('flash' in e['name'].lower()) if variant=='native' else e['name'] in ('_normalized_partition','_lse_merge'))]
            assert kernels,(folder,variant,'no attention kernels found')
            names=sorted(set(e['name'] for e in kernels))
            for name in names:
                selected=[e for e in kernels if e['name']==name]
                traces.append({'batch':b,'context':n,'dim':d,'variant':variant,'kernel':name,
                    'count':len(selected),'median_us':st.median(e['dur'] for e in selected),
                    'grid':selected[0].get('args',{}).get('grid'),'block':selected[0].get('args',{}).get('block')})
    ncu=(ROOT/'results/smalld-ncu.log')
    text=ncu.read_text(errors='replace') if ncu.exists() else ''
    stdout=ROOT/'results/smalld-ncu-stdout.log'
    if stdout.exists():text+='\n'+stdout.read_text(errors='replace')
    exit_file=ROOT/'results/smalld-ncu.exit'
    exit_code=int(exit_file.read_text().strip()) if exit_file.exists() else None
    ncu_status=('counter_permission_denied' if 'ERR_NVGPUCTRPERM' in text else
        'timeout_no_kernels' if exit_code==124 and 'No kernels were profiled' in text else 'inspect_raw_log')
    summary={'cases':raw['cases'],'graph_rows':rows,'host_rows':hosts,'traces':traces,
        'nonconverged':[r for r in rows if not r['converged_5pct']],
        'ncu_status':ncu_status,'ncu_exit_code':exit_code,'kernel_changed':False,'new_model_throughput_claim':False}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
    get=lambda c,v:next(r for r in rows if (r['batch'],r['context'],r['dim'],r['variant'])==(c['batch'],c['context'],c['dim'],v))['primary_us']
    maximum_api_bare_gap=max(abs(get(c,'matched_api')-get(c,'matched_bare'))/get(c,'matched_api') for c in raw['cases'])
    assert all(r['variant']=='merge_only' for r in summary['nonconverged'])
    lines=['# D64/128性能退化的分阶段诊断','',
        f'**结论：优先优化_normalized_partition第一阶段。** 九个形状的API与bare设备区间最大相对差为{maximum_api_bare_gap*100:.3f}%；D64/128完整算子的高耗时与第一阶段时间线相互印证。Python校验及merge不是这轮设备退化的首要优化对象。D64出现编译spill，但D128长上下文没有spill仍明显退化，因此不能把原因简化成spill一项。','',
        '本轮保持生产kernel与分区策略不变，使用Hq32/Hkv8，D64/128各三个代表形状，并加入相同heads的D256控制。它不是Qwen3.5两档模型的D256几何，也不新增端到端性能结论。','',
        '正式区间：5轮随机交错variant/unroll，每轮20次replay，主指标Graph128区间除以算子次数，Graph32用于5%收敛检查。分阶段与bare直接调用复用相同资源和kernel。额外保存包含同步的完整调用wall time、独立CUDA时间线和编译资源。','',
        '| D | B / KV | 原生 μs | API μs | bare μs | 分区 μs | 合并 μs | 原生/API |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    cases=sorted(raw['cases'],key=lambda c:(c['dim'],c['batch'],c['context']))
    for c in cases:
        vals=[get(c,v) for v in ('native','matched_api','matched_bare','partition_only','merge_only')]
        cells=[]
        for value,variant in zip(vals,('native','matched_api','matched_bare','partition_only','merge_only')):
            r=next(r for r in rows if (r['batch'],r['context'],r['dim'],r['variant'])==(c['batch'],c['context'],c['dim'],variant))
            cells.append(f'{value:.3f}'+('†' if not r['converged_5pct'] else ''))
        lines.append(f"| {c['dim']} | {c['batch']} / {c['context']} | "+' | '.join(cells)+f" | {vals[0]/vals[1]:.3f} |")
    lines+=['','![Measured intervals by stage](stages.png)','',f"共450条Graph重复记录、180条完整调用分组（每组20次）、18份独立profiler trace；非收敛配置{len(summary['nonconverged'])}/45。所有9个形状通过FP32参考和bare/API逐位对照，捕获后的完整输出另作检查。native逐位一致情况与max/RMSE见case原始记录。",'',
        '†表示该孤立阶段32/128差异超过5%，其绝对延迟尚未收敛，不能当作精确merge成本。完整、bare、partition计时及独立profiler用于共同判断主开销位置。','',
        '## 独立CUDA时间线与完整调用','',
        '| D | B / KV | 分区kernel μs | 合并kernel μs | 预分配完整wall μs | 含分配完整wall μs |',
        '|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        select=lambda records,v,key:next(r[key] for r in records if (r['batch'],r['context'],r['dim'],r.get('variant'))==(c['batch'],c['context'],c['dim'],v))
        stages={r['kernel']:r for r in traces if (r['batch'],r['context'],r['dim'],r['variant'])==(c['batch'],c['context'],c['dim'],'matched_api')}
        assert all(stages[k]['count']==96 for k in ('_normalized_partition','_lse_merge'))
        lines.append(f"| {c['dim']} | {c['batch']} / {c['context']} | {stages['_normalized_partition']['median_us']:.3f} | {stages['_lse_merge']['median_us']:.3f} | {select(hosts,'matched_preallocated','median_of_repeat_medians_us'):.3f} | {select(hosts,'matched_allocate','median_of_repeat_medians_us'):.3f} |")
    lines+=['','时间线为单独profiling运行的96次算子样本，带profiling开销，不能替换上表正式区间；wall time包含前后同步，不能直接用差值估计精确分配成本。原生时间线中的实际kernel名称、数量与时长保留在summary.json，不假定每种形状都使用两个kernel。','',
        '## 编译资源与计划','',
        '| D | B / KV | splits / tile | 分区program | 每线程寄存器 | 编译spill字段 | PTX KiB | workspace bytes |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        k=c['compiled']['partition'];plan=c['plan']
        lines.append(f"| {c['dim']} | {c['batch']} / {c['context']} | {plan['splits']} / {plan['tile_tokens']} | {c['partition_programs']} | {k['registers_per_thread']} | {k['spill_count']} | {c['partition_ptx_bytes']/1024:.1f} | {c['workspace_bytes']} |")
    lines+=['',
        'n_regs/n_spills来自实际launch后的Triton compiled kernel，PTX及metadata完整保留。它们是静态编译资源，不是动态spill流量或achieved occupancy。相同head配置中GQA group=4，而计算块有16个query行；这提供无效工作线索，但不单独证明它是主要原因。','',
        '## 如何使用这份证据','',
        '先检查相同形状下API与bare是否接近，以及独立时间线中的_normalized_partition与_lse_merge耗时。若完整调用和分区都高、合并低，应优先研究第一阶段；单独阶段的时间不能机械相减并认作精确开销分摊。校验和分配位于Graph捕获前，无法解释replay的设备退化，但会影响完整eager调用。','',
        'D64使用256-token tile，D128使用128-token tile，D256使用64-token tile。第一阶段为保留数值协议，展开了按lane有序的分母累加；更大的tile增加需要保留的中间概率和归约指令。寄存器达到上限或发生spill与这一设计相容，仍须结合硬件指标或控制实验判断因果。D256也可能发生spill，不能把所有D差异都归因于spill；没有spill的D128退化同样需要解释。','',
        f'单独Nsight Compute探测状态：`{ncu_status}`，退出码`{exit_code}`。原始日志为results/smalld-ncu.log及stdout/exit；探测在正式计时完成后运行，不把profiler测得的延迟放入上表。限时未采到kernel不能解释为已证实计数器权限被拒绝。本轮没有有效硬件指标，不报告硬件occupancy、带宽利用率或动态local-memory流量。','',
        '## 优化实验的选择','',
        '优先对第一阶段做一个显式的num_warps=8控制实验，比较相同tile/分区/公式下的编译寄存器与spill、分区延迟和输出；然后研究有序分母代码的临时量生命周期。改变warp数可能改变编译映射与舍入，不能直接替换正式后端。每个候选必须同时检查FP32误差、原生逐位状态和跨页/空行Graph，再决定是否扩展到真实模型。','',
        '暂不优先优化merge或Python校验，不通过改变分区策略来掩盖第一阶段成本。tile变小会改变归约协议，属于另一个显式实验，需要完整的数值回归。','',
        '## 复现与限制','',
        '```bash','source tools/env.sh','export CUDA_VISIBLE_DEVICES=0',
        '/path/to/venv/bin/python tools/diagnose_small_d.py --output results/new-small-d',
        '/path/to/venv/bin/python tools/report_small_d.py --input results/new-small-d --output docs/new-small-d','```','',
        '所选GPU必须空闲，工具建立宿主PID映射并逐轮检查共享，发生共享会中止。未锁频；五轮范围是描述性波动。Graph重复读取同一KV，cache复用与真实生成不同。独立profiler样本也受profiling影响，只用作阶段与执行结构的诊断。完整调用wall time包含同步成本，不能与设备区间混为一谈。','']
    (args.output/'RESULTS.md').write_bytes(('\n'.join(lines)).encode())
    print(json.dumps({'cases':len(cases),'rows':len(raw['rows']),'nonconverged':len(summary['nonconverged']),'ncu_status':ncu_status}))


if __name__=='__main__':main()
