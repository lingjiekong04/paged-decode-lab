"""Report ordered-denominator selection versus the unchanged production partition."""
import json,re,statistics as st
from _common import ROOT


def main():
    bench=ROOT/'results/denom-benchmark-v1';correct=ROOT/'results/denom-correctness-v1'
    data=json.loads((bench/'summary.json').read_text());num=json.loads((correct/'summary.json').read_text())
    if data['rejections']:raise RuntimeError('review rejected candidates before using this report')
    assert data['status']=='completed' and len(data['rows'])==360
    assert num['status']=='passed' and not num['compile_failures']
    log=(ROOT/'results/denom-tests.log').read_text()
    assert re.search(r'7 passed',log) and 'failed' not in log
    rows=[];cases=[]
    for file in sorted(bench.glob('*/case.json')):
        c=json.loads(file.read_text());b,n,d=c['batch'],c['context'],c['dim'];cases.append(c)
        assert c['compiled']['baseline']['plan']==c['compiled']['select']['plan']
        for w in ('baseline','select'):
            k=c['compiled'][w]
            k['shared_bytes']=int(re.search(r'\bshared=(\d+)',k['metadata'])[1])
            k['ptx_bytes']=file.with_name(f'partition-{w}.ptx').stat().st_size
        for stage in ('partition','complete'):
            summary={};paired={}
            for w in ('baseline','select'):
                groups={u:[r for r in data['rows'] if (r['batch'],r['context'],r['dim'],r['stage'],r['variant'],r['unroll'])==(b,n,d,stage,w,u)] for u in (32,128)}
                assert all(len(g)==5 for g in groups.values())
                vals=[r['device_interval_us'] for r in groups[128]];median=st.median(vals)
                delta=abs(st.median(r['device_interval_us'] for r in groups[32])-median)/median
                summary[w]={'median_us':median,'range_us':[min(vals),max(vals)],'convergence_delta':delta,'converged_5pct':delta<=.05}
                paired[w]={r['repeat']:r['device_interval_us'] for r in groups[128]}
            ratios=[paired['baseline'][i]/paired['select'][i] for i in range(5)]
            rows.append({'batch':b,'context':n,'dim':d,'stage':stage,'variants':summary,
                'speedup_baseline_over_select':summary['baseline']['median_us']/summary['select']['median_us'],
                'paired_ratio_median':st.median(ratios),'paired_ratio_range':[min(ratios),max(ratios)],
                'both_converged':all(r['converged_5pct'] for r in summary.values())})
    assert len(cases)==9 and len(rows)==18
    checks=[c for c in num['checks'] if 'fp32_close' in c]
    comparisons=[c for c in num['checks'] if c.get('comparison')=='baseline_vs_select']
    counts={'fp32_and_graph_checks':len(checks),'outputs_bitwise_equal':sum(c['output']['bitwise_equal'] for c in comparisons),
        'partials_bitwise_equal':sum(c['partials']['bitwise_equal'] for c in comparisons),
        'lse_bitwise_equal':sum(c['lse']['bitwise_equal'] for c in comparisons),'pair_checks':len(comparisons),
        'benchmark_native_equal':sum(c['native_equal'] for c in data['checks']),'benchmark_checks':len(data['checks'])}
    result={'rows':rows,'cases':cases,'numerics':counts,'protocol_tests_passed':7,
        'nonconverged':[{'batch':r['batch'],'context':r['context'],'dim':r['dim'],'stage':r['stage'],'variant':w,**s}
            for r in rows for w,s in r['variants'].items() if not s['converged_5pct']],
        'production_changed':False,'model_logits_tested':False,'hardware_counters_obtained':False}
    static=json.loads((ROOT/'results/denom-static-summary.json').read_text())
    result['static_machine_code']=static
    limits=json.loads((ROOT/'results/denom-shared-limit.json').read_text())
    result['static_shared_limit']=limits
    out=ROOT/'docs/denominator';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    lines=['# 有序分母归约：直接块选择实验','',
        '**结论：候选数值通过，但不采用本次改写。** 原实现在每个8-token块上使用masked reduction提取偶数/奇数项；候选改成静态split/transpose选择小块，保留逐lane、逐块的FP32累加次序。PTX文本缩小，但最终stack与局部内存指令增加，不能据此认为临时量生命周期已经改善。','',
        '两条路径的partition和merge均为4 warps，tile、分区规划、QK/PV、exp/log、最终四lane合并和workspace几何固定。实验接口独立于正式后端，不修改默认配置。','',
        '| D | B / KV | 原完整 μs | 候选完整 μs | 原/候选 | 原分区 μs | 候选分区 μs | 均收敛 |',
        '|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in sorted(cases,key=lambda c:(c['dim'],c['batch'],c['context'])):
        get=lambda stage:next(r for r in rows if (r['dim'],r['batch'],r['context'],r['stage'])==(c['dim'],c['batch'],c['context'],stage))
        a,b=get('complete'),get('partition')
        lines.append(f"| {c['dim']} | {c['batch']} / {c['context']} | {a['variants']['baseline']['median_us']:.3f} | {a['variants']['select']['median_us']:.3f} | {a['speedup_baseline_over_select']:.3f} | {b['variants']['baseline']['median_us']:.3f} | {b['variants']['select']['median_us']:.3f} | {'是' if a['both_converged'] and b['both_converged'] else '否'} |")
    lines+=['','比值<1表示候选退化。五次范围和同轮配对比值见summary.json；不跨campaign拼接提速比例。','',
        '![Paired complete intervals](latency.png)','',
        '## 编译资源与最终机器码','',
        '| D | B / KV | regs/thread 原→候选 | spill字段 原→候选 | PTX KiB 原→候选 |',
        '|---:|---:|---:|---:|---:|']
    for c in sorted(cases,key=lambda c:(c['dim'],c['batch'],c['context'])):
        a,b=c['compiled']['baseline'],c['compiled']['select']
        lines.append(f"| {c['dim']} | {c['batch']} / {c['context']} | {a['registers_per_thread']}→{b['registers_per_thread']} | {a['spill_count']}→{b['spill_count']} | {a['ptx_bytes']/1024:.1f}→{b['ptx_bytes']/1024:.1f} |")
    lines+=['',
        'D64、B1、KV8192另外检查了与编译metadata哈希对应的cubin。cuobjdump资源记录为REG 255→32、STACK 152→9760；静态SASS中的LDL指令38→2922、STL指令41→1703。原始资源输出、SASS及cubin哈希保留在results/denom-static-*。这些是静态字段与指令站点数，不是动态spill次数、执行流量或硬件occupancy。', '',
        'cuobjdump的静态SHARED字段为0，但launch所需动态shared由Tritonmetadata记录，不能解释成无需shared memory。PTX中未出现ld.local/st.local，不代表最终机器码没有局部内存访问；本例需要检查ptxas之后的SASS。PTX文本更短、虚拟寄存器更少、实际REG字段更低，都不能单独作为优化成功标准。','',
        f"另外在同型号GPU1上通过CUDA Runtime查询静态上限：每SM shared {limits['shared_bytes_per_sm']}字节。D64两条路径的动态shared均为85248字节，因此仅按shared容量计算的并驻CTA上界为1；降低REG字段并不会自动提高这个上界。这不是实际occupancy测量，其他资源和调度条件仍可能进一步限制运行。查询属性编号来自本环境CUDA头文件，原始值见denom-shared-limit.json。",'',
        '当前证据支持拒绝这个选择器实现，但不足以精确区分layout转换和后端寄存器分配策略各自的因果份额。本轮没有取得有效硬件计数器，不报告带宽利用率或achieved occupancy。','',
        '## 数值验收','',
        f"7项新增测试通过：4项控制条件测试、3项GPU选择器重建顺序测试。D64/128/256小规模预检的输出、partials和LSE与原实现逐位一致。完整性能矩阵{counts['benchmark_checks']}次FP32检查，其中{counts['benchmark_native_equal']}次输出与原生逐位相同。",'',
        f"边界矩阵覆盖D64/128/256、GQA group 1/4/16、随机物理页、初始-1补齐与NaN尾部、0/1/255/256/257/1023/1024/1025长度、增长到2051、KV更新、页映射翻转和全空行。共{counts['fp32_and_graph_checks']}次输出通过atol=rtol=2e-2及Graph/eager一致性检查；{counts['pair_checks']}组原/候选对照中output {counts['outputs_bitwise_equal']}、partials {counts['partials_bitwise_equal']}、LSE {counts['lse_bitwise_equal']}组逐位一致。原实现适配器同时对照正式matched_decode。",'',
        '测试只证明所列算子场景，不外推为任意输入或模型等价。本轮没有将候选接入模型，也没有新增逐步模型logits或端到端吞吐结果。','',
        '## 实现与下一步','',
        '固定Triton3.2没有tl.gather；候选使用有限层级的split/transpose选择函数，避免递归JIT依赖。首次预检遇到递归依赖哈希和API写法兼容问题，失败源码/日志作为setup诊断保留，均发生在正式计时之前。最终候选源码哈希随实验记录固定。','',
        '下一步先确认概率数据在各线程中的原始MMA分布，再设计直接读取线程所持小片段的选择器，避免为抽取小块反复改变整个tile布局。另一个更小的控制实验是仅将V加载推迟到PV之前，观察编译器是否仍提前调度，以及K/V/概率临时量的资源重叠是否改变。保持四个lane内部的偶/奇/块次序与最终(s0+s2)+(s1+s3)不变；不能用普通tl.sum替换有序归约后直接宣称数值协议保持一致。','',
        '## 复现与计时边界','',
        f"9形状、360条重复记录，{len(result['nonconverged'])}/36个配置未通过32/128的5%收敛检查。5轮随机交错variant/stage/unroll，每轮20次replay；主指标固定Graph128区间除以算子次数。JIT、预热、捕获、正确性检查和静态机器码读取在正式计时区间之外。所选GPU建立宿主PID映射并逐轮检查共享；正确性使用另一张GPU，未宣称整台主机独占。",'',
        'Graph重复读取固定输入和KV，区间包含graph节点间隙，不是纯kernel busy time或真实连续生成延迟。未锁频，五轮范围为描述性波动，不是置信区间。','',
        '```bash','source tools/env.sh','export CUDA_VISIBLE_DEVICES=0',
        '/path/to/venv/bin/python tools/smoke_denominator.py --output results/new-denom-smoke.json',
        '/path/to/venv/bin/python tools/ablate_denominator.py --task correctness --output results/new-denom-correctness',
        '/path/to/venv/bin/python tools/ablate_denominator.py --task benchmark --output results/new-denom-benchmark','```','',
        '报告生成器默认读取本次固定结果路径；输出目录必须为新目录。所有生产kernel和历史实验保持原样。','']
    (out/'RESULTS.md').write_bytes(('\n'.join(lines)).encode())
    print(json.dumps({'cases':len(cases),'records':len(data['rows']),'numerics':counts,'nonconverged':len(result['nonconverged'])}))


if __name__=='__main__':main()

