"""Explain short-context overhead using controlled unroll and host-only probes."""
from pathlib import Path
import json,statistics as st,math
from _common import ROOT


def main():
    folder=ROOT/'results/shortdiag-20261103'
    d=json.loads((folder/'summary.json').read_text())
    rows=[]; traces=[]; hosts=[]
    for model in ('small','large'):
        for n in (512,8192):
            for variant in ('native','matched_api','matched_bare','partition_only','merge_only'):
                for unroll in (1,8,32):
                    selected=[r for r in d['rows'] if (r['model_geometry'],r['context'],r['variant'],r['unroll'])==(model,n,variant,unroll)]
                    assert len(selected)==5
                    rows.append({'model':model,'context':n,'variant':variant,'unroll':unroll,
                        'interval_us':st.median(r['device_interval_us'] for r in selected),
                        'interval_range':[min(r['device_interval_us'] for r in selected),max(r['device_interval_us'] for r in selected)],
                        'host_submission_us_per_replay':st.median(r['host_submission_us_per_replay'] for r in selected)})
            for variant in ('native','matched_api'):
                for mode in ('eager','graph1','graph32'):
                    path=folder/f'{model}-l{n}-{variant}-{mode}-trace.json'
                    raw=json.loads(path.read_text())['traceEvents']
                    kernels=sorted([e for e in raw if e.get('cat')=='kernel' and
                        ('flash_fwd_splitkv' in e['name'] if variant=='native' else e['name'] in ('_normalized_partition','_lse_merge'))],key=lambda e:e['ts'])
                    assert len(kernels)==(320 if mode=='graph32' else 10),(path,len(kernels))
                    traces.append({'model':model,'context':n,'variant':variant,'mode':mode,'kernel_count':len(kernels),
                        'first_kernel_us':st.median(e['dur'] for e in kernels[::2]),
                        'merge_kernel_us':st.median(e['dur'] for e in kernels[1::2]),
                        'interstage_gap_us':st.median(b['ts']-a['ts']-a['dur'] for a,b in zip(kernels[::2],kernels[1::2])),
                        'source':path.relative_to(ROOT).as_posix()})
    host=json.loads((ROOT/'results/shortdiag-host/summary.json').read_text())
    for h in (8,16):
        for n in (512,8192):
            for v in ('validate_only','allocate_only'):
                hosts.append({'query_heads':h,'context':n,'variant':v,'cpu_us':st.median(r['cpu_wall_us'] for r in host['rows'] if (r['query_heads'],r['context'],r['variant'])==(h,n,v))})
    out=ROOT/'docs/short-overhead';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_bytes((json.dumps({'graph_rows':rows,'traces':traces,'host_rows':hosts,'cases':d['cases']},indent=2)+'\n').encode())
    get=lambda m,n,v,u:next(r for r in rows if (r['model'],r['context'],r['variant'],r['unroll'])==(m,n,v,u))['interval_us']
    lines=['# 短上下文额外开销诊断','',
        '**结论：先修正算子基准的提交方式，不根据旧36对25μs结果修改attention kernel。** 本轮没有修改kernel、分区规划或模型运算。受控补测未复现此前短KV的新路径退化；一张Graph只执行一次算子的测量会明显受到host提交速率影响。','',
        '旧报告是实际观测记录，继续保留；但不能把那些包含提交空隙的区间直接归因为kernel计算更慢。之前的计时已标明包含执行间隙，本轮进一步识别了这一因素。','',
        '## 受控实验','',
        '同一张A100、固定B1/D256，两档模型几何，KV512及8192对照。native和matched均预热；每张Graph分别包含1/8/32次相同算子，50次replay构成一次测量，独立重复5次，每个repeat随机交错全部variant与unroll顺序。区间除以实际算子次数；另外保留host提交耗时、分阶段Graph和独立CPU/CUDA时间线。','',
        '| 几何 / KV | 原生 Graph1 μs | 新路径 Graph1 μs | 原生 Graph32/32 μs | 新路径 Graph32/32 μs |',
        '|---|---:|---:|---:|---:|']
    for m in ('small','large'):
        for n in (512,8192):
            lines.append(f"| {'0.8B' if m=='small' else '9B'} / {n} | {get(m,n,'native',1):.3f} | {get(m,n,'matched_api',1):.3f} | {get(m,n,'native',32):.3f} | {get(m,n,'matched_api',32):.3f} |")
    lines+=['','![Graph unroll convergence](convergence.png)','','## 各项开销怎么解释','',
        '1. **不是新路径独有的第二次launch。** 时间线确认原生也运行split-KV第一阶段和combine，二者均为两个attention kernel。不能使用“原生一次、新路径两次”解释差距。',
        '2. **单次replay的区间不等于设备忙碌时间。** 即便只捕获约2μs的merge kernel，Graph1区间仍可能十几μs；Graph32把host提交摊薄后才接近设备执行成本。时间线与独立阶段计时只作诊断，不从不同实验区间机械相减来估计固定launch成本。',
        '3. **校验和workspace分配不能解释Graph replay的差异。** 两者都在捕获前执行；matched API与直接launch两个相同kernel的bare路径，在Graph32下的计时接近。分配会影响完整eager调用，不能加进Graph replay成本。','',
        '| 几何 / KV512 | 分区阶段 Graph32 μs | 合并阶段 Graph32 μs | API完整 Graph32 μs | bare完整 Graph32 μs |',
        '|---|---:|---:|---:|---:|']
    for m in ('small','large'):
        lines.append(f"| {'0.8B' if m=='small' else '9B'} | {get(m,512,'partition_only',32):.3f} | {get(m,512,'merge_only',32):.3f} | {get(m,512,'matched_api',32):.3f} | {get(m,512,'matched_bare',32):.3f} |")
    lines+=['','| 几何 / KV512 | 仅元数据校验 CPU μs | 仅缓存分配 CPU μs |','|---|---:|---:|']
    for h in (8,16):
        val={r['variant']:r['cpu_us'] for r in hosts if r['query_heads']==h and r['context']==512}
        lines.append(f"| Hq{h} | {val['validate_only']:.3f} | {val['allocate_only']:.3f} |")
    lines+=['',
        'CPU专测每repeat调用1000次，5次重复，计时前同步一次，计时内部没有kernel或逐次同步。另有20次逐调用同步wall样本保留在原始host文件；它们包含同步成本，不能当作纯CPU函数开销。','',
        '## 接下来改哪里','',
        '- 优先改后续正式算子基准：至少使用Graph内多次执行并报告1/8/32的收敛情况，随机交错后端，明确区分host提供速率、设备区间和独立profiler busy time。不要把“Graph32”写成新的推理kernel或模型加速。',
        '- 若继续优化设备代码，先研究分区阶段：KV512下约8.2μs，合并约2.0μs。small第一阶段仅16个program、large仅32个，均低于108个SM；这提供低并行度的线索，但未取得硬件计数器，不能认定occupancy或寄存器压力已经确诊。',
        '- 暂不改分区数和归约顺序；它们关系到已验证的数值协议。减少merge调用或改tile需要另立正确性和逐步logits回归。','',
        '## 限制与证据','',
        '未锁频，未测硬件occupancy。Graph内重复使用相同KV/元数据，旨在诊断稳定算子吞吐，不能代替实际逐token模型decode。不同运行的GPU时钟、cache状态与host调度也可能影响旧/新数值；证据支持存在host提交上限以及旧退化未复现，不足以把旧差值精确拆为某几微秒。','',
        '原始Chrome trace包含profiler自身初始化kernel；汇总只选择两个attention kernel，保留完整trace供复核。profiling与正式计时分开。全部原始行、5次范围、trace分阶段时长/间隙及CPU成本见summary.json与results/shortdiag-*。','',
        '```bash','source tools/env.sh',
        '.venv35/bin/python tools/diagnose_short_overhead.py --output results/new-shortdiag',
        '.venv35/bin/python tools/diagnose_host_overhead.py --output results/new-shortdiag-host',
        '.venv35/bin/python tools/report_short_overhead.py','```','',
        '报告生成器默认读取交付包内固定命名的本轮原始结果。原始SHA256索引为SHORTDIAG_EVIDENCE_INDEX.json和SHORTDIAG_HOST_EVIDENCE_INDEX.json。','']
    (out/'RESULTS.md').write_bytes(('\n'.join(lines)).encode())
    print('Generated graph convergence, stage, timeline and CPU summaries')


if __name__=='__main__': main()
