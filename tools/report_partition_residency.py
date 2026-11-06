"""Validate the CTA-grid / unused-shared-padding causal diagnostic."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as stats


def describe(values):
    return {'median':stats.median(values),'min':min(values),'max':max(values),'raw':values}


def aggregate(data):
    if data.get('status')!='passed':raise ValueError('incomplete campaign')
    expected={(n,v,g,u,r) for n in (512,2048) for v in ('baseline','baseline_padded','shared')
              for g in (18,24,27,28,32,33,36) for u in (32,128) for r in range(5)}
    values={}
    for row in data['rows']:
        key=tuple(row[k] for k in ('input_tokens','variant','grid_parts','unroll','repeat'))
        if key not in expected or key in values:raise ValueError('unexpected or duplicate record')
        value=row['us_per_call']
        if not math.isfinite(value) or value<=0:raise ValueError('invalid timing')
        if row['context']!=key[0]+16 or row['total_ctas']!=4*key[2]:raise ValueError('invalid geometry')
        if row['active_ctas']!=4*min(key[2],math.ceil((key[0]+16)/64)):raise ValueError('invalid active CTA count')
        values[key]=value
    if set(values)!=expected:raise ValueError('incomplete timing matrix')
    if len(data['checks'])!=2 or {c['input_tokens'] for c in data['checks']}!={512,2048}:
        raise ValueError('missing numerical checks')
    for check in data['checks']:
        if not check['native_bytes_equal'] or not check['padded_partials_bytes_equal']:
            raise ValueError('numerical control failed')
        r=check['resources']
        for v in ('baseline','baseline_padded','shared'):
            if r[v]['registers_per_thread']!=r['baseline']['registers_per_thread']:
                raise ValueError('register geometry changed')
        if r['baseline_padded']['dynamic_shared_bytes']+r['baseline_padded']['static_shared_bytes'] != \
           r['shared']['dynamic_shared_bytes']+r['shared']['static_shared_bytes']:
            raise ValueError('padding does not match shared footprint')
    groups=[]
    for n in (512,2048):
        for g in (18,24,27,28,32,33,36):
            variants={}
            for v in ('baseline','baseline_padded','shared'):
                timings={str(u):describe([values[n,v,g,u,r] for r in range(5)]) for u in (32,128)}
                delta=timings['32']['median']/timings['128']['median']-1
                variants[v]={'timings_us':timings,'unroll_relative_delta':delta,'converged_3pct':abs(delta)<=.03}
            ratios={name:describe([values[n,a,g,128,r]/values[n,b,g,128,r] for r in range(5)])
                    for name,a,b in (('baseline_over_shared','baseline','shared'),
                                    ('padded_over_baseline','baseline_padded','baseline'),
                                    ('padded_over_shared','baseline_padded','shared'))}
            groups.append({'input_tokens':n,'context':n+16,'parts':g,'total_ctas':4*g,
                           'active_ctas':4*min(g,math.ceil((n+16)/64)),'variants':variants,'ratios':ratios})
    return groups


def ncu_status(root):
    result=[]
    for job in sorted(root.rglob('residency-ncu-*.job.json')):
        entry=json.loads(job.read_text());tag=entry['tag']
        path=job.parent/(tag+'.csv')
        text=path.read_text(errors='replace') if path.exists() else ''
        log=(job.parent/(tag+'.log')).read_text(errors='replace')
        if 'ERR_NVGPUCTRPERM' in text+log:status='counter_permission_denied'
        elif entry['timeout']:status='timeout_no_valid_counters'
        elif entry['exit_code']!=0:status='failed'
        else:status='completed'
        metrics=[]
        lines=text.splitlines()
        start=next((i for i,line in enumerate(lines) if '"Metric Name"' in line and '"Metric Value"' in line),None)
        if start is not None:
            metrics=list(csv.DictReader(lines[start:]))
        if status=='completed' and not metrics:status='completed_without_metrics'
        result.append({'tag':tag,'status':status,'job':entry,'metrics':metrics})
    return result


def plot(groups,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,n in zip(axes,(512,2048)):
        rows=[g for g in groups if g['input_tokens']==n]
        for v,label,color in [('baseline','Original (limit: 2 CTAs/SM)','#2577a3'),
                              ('baseline_padded','Original +4352 B (limit: 1 CTA/SM)','#bb782c'),
                              ('shared','Shared (limit: 1 CTA/SM)','#2b8b68')]:
            xs=[g['total_ctas'] for g in rows]
            obs=[g['variants'][v]['timings_us']['128'] for g in rows]
            ys=[o['median'] for o in obs]
            ax.errorbar(xs,ys,yerr=[[o['median']-o['min'] for o in obs],[o['max']-o['median'] for o in obs]],
                        marker='o',capsize=3,label=label,color=color)
        ax.axvline(108,linestyle='--',color='#7b8490',linewidth=1,label='108 SMs')
        ax.set_title(f"Context {n+16}: fixed cubins, grid sweep")
        ax.set_xlabel('Launched CTAs (4 KV heads × grid parts)')
        ax.set_ylabel('Partition interval per call (µs)')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('A100 D256: unused shared reservation control | five-round min–max')
    fig.savefig(output/'residency.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--benchmark',default='residency-benchmark-v3')
    args=p.parse_args()
    raw=json.loads((args.root/args.benchmark/'summary.json').read_text())
    monitor=json.loads((args.root/args.benchmark/'gpu-monitor.json').read_text())
    if monitor['violations'] or not monitor['rows']:raise ValueError('invalid GPU isolation')
    resources=json.loads((args.root/'residency-resources-v1/resources.json').read_text())
    groups=aggregate(raw)
    profiles=ncu_status(args.root)
    summary={'groups':groups,'resources':resources,'ncu':profiles,'monitor_samples':len(monitor['rows']),
             'nonconverged':[{k:g[k] for k in ('input_tokens','parts')}|{'variant':v}
                             for g in groups for v,t in g['variants'].items() if not t['converged_3pct']],
             'production_changed':False,'new_model_throughput_claim':False}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(groups,args.output)
    long=next(g for g in groups if g['context']==2064 and g['parts']==36)
    before=next(g for g in groups if g['context']==2064 and g['parts']==27)
    after=next(g for g in groups if g['context']==2064 and g['parts']==28)
    long_times={v:x['timings_us']['128']['median'] for v,x in long['variants'].items()}
    penalty=1/long['ratios']['baseline_over_shared']['median']-1
    padding=long['ratios']['padded_over_baseline']['median']-1
    lines=['# 9B 长上下文 partition 驻留诊断','',
        '针对 batch=1、16 query heads／4 KV heads、D256、容量 2112、36 partitions、TILE=64。',
        '基于固定 revision Qwen3.5-9B 的真实 layer 3 Q/K/V，实际上下文 528／2064；其余层的退化证据见上一轮分段报告。','',
        '## 结论','',
        f"长上下文、完整 grid 的 partition 延迟：原版 {long_times['baseline']:.4f} µs，原版 padding {long_times['baseline_padded']:.4f} µs，shared {long_times['shared']:.4f} µs。shared 比原版慢约 {penalty:.2%}；原版只增加预留、不改变指令，延迟就增加约 {padding:.2%}。",'',
        f"百分比取同轮配对比值的五轮中位数。完整 grid 的原版/shared 比值范围为 {long['ratios']['baseline_over_shared']['min']:.5f}–{long['ratios']['baseline_over_shared']['max']:.5f}，padding/原版范围为 {long['ratios']['padded_over_baseline']['min']:.5f}–{long['ratios']['padded_over_baseline']['max']:.5f}；五轮方向一致。",'',
        f"保持同一 cubin 扫描 grid：108 个非空 CTA（27 partitions）时，原版/shared 比值为 {before['ratios']['baseline_over_shared']['median']:.4f}，原版 padding/原版为 {before['ratios']['padded_over_baseline']['median']:.4f}；112 个非空 CTA 时，两项分别变为 {after['ratios']['baseline_over_shared']['median']:.4f} 和 {after['ratios']['padded_over_baseline']['median']:.4f}。这个跨越与 108 个 SM 的设备规模及 2→1 CTA/SM 的驻留上限一致。",'',
        '短上下文虽发射同样的完整 grid，只有 36 个非空 CTA，padding 没有产生长上下文的惩罚，shared 仍更快。静态资源、等量预留控制和非空 CTA 扫描共同支持：新增 shared memory 触发驻留门槛，是此次退化的主要可复现因素。',
        '在同为 1 CTA/SM 上限的配置下，shared 又比 padded 原版更快，说明实现收益与资源代价同时存在。没有硬件计数器时，不把它写成 achieved occupancy 或动态 spill 流量已被直接测得，也不声称排除了所有调度／缓存影响。','',
        '下一项优化应先减少 shared 总占用，使驱动重新允许 2 CTA/SM，再验证数值、短／长上下文和模型回归；本轮没有修改生产 kernel 或默认后端。','',
        '## 编译资源与驱动驻留上限','',
        '| 版本 | 寄存器/线程 | Triton spill 字段 | local bytes/线程 | dynamic shared | static shared | 驱动预留/CTA | 最大 CTA/SM |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v,r in resources['variants'].items():
        lines.append(f"| {v} | {r['registers_per_thread']} | {r['triton_spill_count']} | {r['local_bytes_per_thread']} | {r['dynamic_shared_bytes']} | {r['static_shared_bytes']} | {r['reserved_shared_bytes_per_block']} | {r['driver_max_active_blocks_per_sm']} |")
    d=resources['device']
    lines+=['',f"设备有 {d['sm_count']} 个 SM；每 SM shared memory 为 {d['shared_bytes_per_sm']} 字节，寄存器为 {d['registers_per_sm']} 个。",'',
      '每 CTA 加上驱动预留后，原版需要 79232+1024=80256 字节，两个为 160512；shared 需要 79232+4352+1024=84608 字节，两个为 169216，超过 167936。',
      '实际驱动 occupancy API 分别返回 2／1 CTA/SM。仅看 Triton dynamic shared 字段会遗漏静态 shared 和驱动预留。',
      '寄存器数相同；spill/local 字段是静态编译资源，不等于运行时 local-memory 流量。驱动 occupancy 结果是可驻留上限，不是 achieved occupancy。','',
      '## 不改变指令的 padding 对照','',
      'baseline_padded 使用同一个原版 compiled kernel，仅把 launch 的 dynamic shared 从 79232 提高为 83584，增加的 4352 字节不参与计算。',
      '它与 shared 的总 shared 预留相同，驱动驻留上限同为 1 CTA/SM；寄存器、输入、线程数、PTX/cubin 与数值指令不变。',
      '所有完整 grid 的输出先与保存的 native 输出字节对齐；padding 前后 partials、LSE 也字节对齐。','',
      '固定 cubin，仅截断 grid 的 partition 维为 18／24／27／28／32／33／36。截断 grid 只验证已启动分区，未启动区域保持 NaN；这些控制不产生完整 attention 输出。',
      '每项随机顺序独立重复 5 轮；分别捕获 32／128 次 partition 调用，各 replay 20 次，正式主表使用 unroll=128；全部波动与收敛检查保留。','',
      '![驻留控制](residency.png)','',
      '| 上下文 | grid parts | 总/非空 CTA | 原版 µs | 原版 padding µs | shared µs | padding/原版 | 原版/shared |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        t={v:x['timings_us']['128']['median'] for v,x in g['variants'].items()}
        lines.append(f"| {g['context']} | {g['parts']} | {g['total_ctas']}/{g['active_ctas']} | {t['baseline']:.4f} | {t['baseline_padded']:.4f} | {t['shared']:.4f} | {g['ratios']['padded_over_baseline']['median']:.4f} | {g['ratios']['baseline_over_shared']['median']:.4f} |")
    lines+=['',f"unroll 32／128 中位数相对差异超过 3% 的控制数：{len(summary['nonconverged'])}，详见 summary.json。",'',
      '第 0 轮部分控制存在较大尾值，全部保留，没有删除或降权；图中误差线为五轮 min–max。unroll 收敛只比较各模式的五轮中位数，不表示每次计时都无波动。未对尾值作硬件归因。','',
      '## 硬件计数器状态','']
    for profile in profiles:
        lines.append(f"- `{profile['tag']}`：`{profile['status']}`；原始 CSV、stdout 与 job.json 均保留。")
    lines+=['', '超时且未捕获 kernel 不等于已证实权限拒绝；未取得有效计数器时，不报告 achieved occupancy、动态 spill 流量或带宽利用率。',
      '最小直接 cubin 目标使用零值 Q/K/V，只用于相同 cubin／grid 的计数器诊断；正式延迟使用真实模型张量，二者不混用。','',
      '## 复现','', '```bash',
      'CUDA_VISIBLE_DEVICES=0 python tools/diagnose_partition_residency.py --task resources --output results/residency-resources-v1',
      'CUDA_VISIBLE_DEVICES=0 python tools/diagnose_partition_residency.py --task snapshot --model models/Qwen3.5-9B --output results/residency-snapshots-v2',
      'CUDA_VISIBLE_DEVICES=0 python tools/diagnose_partition_residency.py --task benchmark --snapshots results/residency-snapshots-v2 --output /tmp/residency-benchmark-v3',
      '# 完成后立即将整个输出归档到 results/residency-benchmark-v3，再生成报告。',
      'python tools/report_partition_residency.py --root results --output docs/partition-residency',
      '```','',
      '仅使用空闲 GPU，过程监控其他计算进程；不替换生产默认后端，不把分区控制的延迟解释为模型吞吐。',
      '首次 snapshot 因存储等待超出时限，重试使用新输出目录，原始失败记录保留。','',
      '首次 benchmark 卡在项目目录元数据写入；后续 /tmp 重试在平台重建时丢失，未计入正式结果。最终 v3 复用保存的真实张量，在空闲 GPU 0 上执行相同冻结脚本；每轮 event 完成、GPU guard 通过后写持久 ZIP 检查点，磁盘写入不在计时区间内。GPU UUID 与源文件哈希保留在 arguments.json。','',
      'Nsight 的 Python、stdlib 直接 cubin 与 host-only C driver 目标均未取得有效计数器。C 目标只加载 Triton cubin，没有实现新的 CUDA kernel；无 profiler 时同一目标正常完成，带 profiler 时限时未捕获 kernel。其源码、编译日志及 plain 日志均保留。','',
      '驱动字段及 occupancy API 语义参考：[CUDA Driver API](https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_Driver_API.pdf)。','']
    (args.output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'groups':len(groups),'nonconverged':len(summary['nonconverged'])}))


if __name__=='__main__':main()
