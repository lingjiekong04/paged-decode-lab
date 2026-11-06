"""Report paired warp-count measurements and numerical gates without selecting a backend."""
import json,re,statistics as st
from _common import ROOT


def main():
    bench=ROOT/'results/warps-benchmark-v1';correct=ROOT/'results/warps-correctness-v1'
    data=json.loads((bench/'summary.json').read_text());num=json.loads((correct/'summary.json').read_text())
    if data['rejections']:raise RuntimeError('review rejected candidates before using this report')
    assert data['status']=='completed' and len(data['rows'])==360
    assert num['status']=='passed' and not num['compile_failures']
    log=(ROOT/'results/warps-protocol-tests.log').read_text()
    assert re.search(r'6 passed',log) and 'failed' not in log
    rows=[];cases=[]
    for file in sorted(bench.glob('*/case.json')):
        c=json.loads(file.read_text());b,n,d=c['batch'],c['context'],c['dim'];cases.append(c)
        assert c['compiled']['4']['plan']==c['compiled']['8']['plan']
        for w in (4,8):
            k=c['compiled'][str(w)]
            k['shared_bytes']=int(re.search(r'\bshared=(\d+)',k['metadata'])[1])
            k['ptx_bytes']=file.with_name(f'partition-w{w}.ptx').stat().st_size
        for stage in ('partition','complete'):
            summary={};paired={}
            for w in (4,8):
                groups={u:[r for r in data['rows'] if (r['batch'],r['context'],r['dim'],r['stage'],r['warps'],r['unroll'])==(b,n,d,stage,w,u)] for u in (32,128)}
                assert all(len(g)==5 for g in groups.values())
                vals=[r['device_interval_us'] for r in groups[128]];median=st.median(vals)
                delta=abs(st.median(r['device_interval_us'] for r in groups[32])-median)/median
                summary[w]={'median_us':median,'range_us':[min(vals),max(vals)],'convergence_delta':delta,'converged_5pct':delta<=.05}
                paired[w]={r['repeat']:r['device_interval_us'] for r in groups[128]}
            ratios=[paired[4][i]/paired[8][i] for i in range(5)]
            rows.append({'batch':b,'context':n,'dim':d,'stage':stage,'warps':summary,
                'speedup_4_over_8':summary[4]['median_us']/summary[8]['median_us'],
                'paired_ratio_median':st.median(ratios),'paired_ratio_range':[min(ratios),max(ratios)],
                'both_converged':all(r['converged_5pct'] for r in summary.values())})
    assert len(cases)==9 and len(rows)==18
    checks=[c for c in num['checks'] if 'fp32_close' in c]
    comparisons=[c for c in num['checks'] if c.get('comparison')=='4_vs_8']
    counts={'fp32_and_graph_checks':len(checks),'outputs_bitwise_equal':sum(c['output']['bitwise_equal'] for c in comparisons),
        'partials_bitwise_equal':sum(c['partials']['bitwise_equal'] for c in comparisons),
        'lse_bitwise_equal':sum(c['lse']['bitwise_equal'] for c in comparisons),'pair_checks':len(comparisons),
        'benchmark_native_equal':sum(c['native_equal'] for c in data['checks']),'benchmark_checks':len(data['checks'])}
    result={'rows':rows,'cases':cases,'numerics':counts,'protocol_tests_passed':6,
        'nonconverged':[{'batch':r['batch'],'context':r['context'],'dim':r['dim'],'stage':r['stage'],'warps':w,**s}
            for r in rows for w,s in r['warps'].items() if not s['converged_5pct']],
        'production_changed':False,'model_logits_tested':False,'hardware_counters_obtained':False}
    out=ROOT/'docs/warp-ablation';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    lines=['# Partition num_warps=4/8控制实验','',
        '**结论：不能把正式后端统一切换到8 warps。** 相同输入、tile、分区和公式下，8 warps可消除本次编译spill，但延迟有收益也有退化；每线程寄存器下降并不自动代表每个CTA的总资源需求下降。结果限定在本次配对实验，不据此自动选择后端。','',
        '只改变第一阶段_normalized_partition的num_warps，第二阶段_lse_merge始终为4。9个形状使用Hq32/Hkv8，D64/128/256，各包含B1/L512、B1/L8192、B32/L8192。分区规划、workspace几何、输入与softmax scale均相同；使用不同输出缓冲区避免互相覆盖。','',
        '| D | B / KV | 完整4 μs | 完整8 μs | 4/8比值 | 分区4 μs | 分区8 μs | 均收敛 |',
        '|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in sorted(cases,key=lambda c:(c['dim'],c['batch'],c['context'])):
        get=lambda stage:next(r for r in rows if (r['dim'],r['batch'],r['context'],r['stage'])==(c['dim'],c['batch'],c['context'],stage))
        full,part=get('complete'),get('partition')
        lines.append(f"| {c['dim']} | {c['batch']} / {c['context']} | {full['warps'][4]['median_us']:.3f} | {full['warps'][8]['median_us']:.3f} | {full['speedup_4_over_8']:.3f} | {part['warps'][4]['median_us']:.3f} | {part['warps'][8]['median_us']:.3f} | {'是' if full['both_converged'] and part['both_converged'] else '否，见摘要'} |")
    lines+=['','比值>1表示8 warps更快，<1表示退化。五轮范围与同轮配对比值全部保留在summary.json；不把几乎重叠的范围解释为稳定收益。','',
        '![Paired complete operator intervals](latency.png)','',
        '## 编译资源','',
        '| D | B / KV | regs/thread 4→8 | spill字段 4→8 | shared bytes 4→8 |',
        '|---:|---:|---:|---:|---:|']
    for c in sorted(cases,key=lambda c:(c['dim'],c['batch'],c['context'])):
        a,b=c['compiled']['4'],c['compiled']['8']
        lines.append(f"| {c['dim']} | {c['batch']} / {c['context']} | {a['registers_per_thread']}→{b['registers_per_thread']} | {a['spill_count']}→{b['spill_count']} | {a['shared_bytes']}→{b['shared_bytes']} |")
    lines+=['',
        '4/8 warps分别对应128/256个线程。需要结合线程数、shared memory及分配粒度考察CTA资源；这里的n_regs/n_spills为编译字段，不是动态local-memory流量、achieved occupancy或带宽利用率。PTX与编译metadata均保留，本轮没有有效硬件计数器。','',
        '## 数值与边界','',
        f"新增6项CPU控制条件测试通过，确认只有partition warps变化，merge配置和计划保持固定。性能矩阵{counts['benchmark_checks']}次FP32检查；其中{counts['benchmark_native_equal']}次输出与原生逐位相同。",'',
        f"边界矩阵包括D64/128/256、GQA group 1/4/16、随机物理页、0/1/255/256/257/1023/1024/1025长度及增长到2051。初始无效table=-1、未使用KV尾部填NaN；随后更新Q/K/V、翻转物理页映射、跨页增长和全空行，重复Graph replay。共{counts['fp32_and_graph_checks']}次输出通过FP32容差和Graph/eager一致性检查；{counts['pair_checks']}组4/8对照中，output {counts['outputs_bitwise_equal']}、partials {counts['partials_bitwise_equal']}、LSE {counts['lse_bitwise_equal']}组逐位一致。",'',
        'FP32容差保持atol=rtol=2e-2。4-warps实验适配器另外与正式matched_decode逐位对照。这些结果限定于算子与边界矩阵；本轮没有把8-warps接入模型，也没有新增逐步模型logits或端到端吞吐结论。','',
        '## 下一步选择','',
        '保留正式4-warps路径。D64可作为8-warps专门候选继续扩大数值及形状回归，但当前D128/D256的退化阻止统一替换。更值得继续研究有序分母归约产生的中间量生命周期与指令展开；目标是在保留数值协议的同时减少资源压力。改变tile或分区数必须单独立项，不能与本轮结果混为同一控制变量。','',
        '## 协议与复现','',
        '5轮随机交错warp数、stage与unroll；每轮20次replay，主指标Graph128区间除以算子次数，Graph32用于5%收敛检查。共360条重复记录。JIT、warmup、capture和正确性检查都在正式计时之外；GPU共享检测在逐轮前后执行。正确性检查使用另一张GPU，环境记录保留各自设备身份，未宣称整台主机独占。未锁频，五轮范围不是置信区间。','',
        'Graph重复读取相同输入和KV，测得的是设备区间，包含graph节点间隙，不能称为纯kernel busy time或实际连续生成。新旧campaign不交叉拼接提速比例；本表只比较同一轮的4/8。','',
        '```bash','source tools/env.sh','export CUDA_VISIBLE_DEVICES=0',
        '/path/to/venv/bin/python tools/ablate_warps.py --task benchmark --output results/new-warps-benchmark',
        '/path/to/venv/bin/python tools/ablate_warps.py --task correctness --output results/new-warps-correctness','```','',
        '报告生成器默认读取results/warps-benchmark-v1和warps-correctness-v1；改名复测时应保留独立原始目录，并调整报告输入或在新的源码副本中复现固定目录。生产kernel、默认配置及历史报告均保持原样。','']
    (out/'RESULTS.md').write_bytes(('\n'.join(lines)).encode())
    print(json.dumps({'cases':len(cases),'rows':len(data['rows']),'numerics':counts,'nonconverged':len(result['nonconverged'])}))


if __name__=='__main__':main()
