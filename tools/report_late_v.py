"""Report V-load placement versus the unchanged production partition."""
import json,re,statistics as st
from _common import ROOT


def main():
    bench=ROOT/'results/latev-benchmark-v1';correct=ROOT/'results/latev-correctness-v1'
    data=json.loads((bench/'summary.json').read_text());num=json.loads((correct/'summary.json').read_text())
    if data['rejections']:raise RuntimeError('review rejected candidates before using this report')
    assert data['status']=='completed' and len(data['rows'])==360
    assert num['status']=='passed' and not num['compile_failures']
    log=(ROOT/'results/latev-tests.log').read_text()
    assert re.search(r'6 passed',log) and 'failed' not in log
    rows=[];cases=[]
    for file in sorted(bench.glob('*/case.json')):
        c=json.loads(file.read_text());b,n,d=c['batch'],c['context'],c['dim'];cases.append(c)
        assert c['compiled']['baseline']['plan']==c['compiled']['late_v']['plan']
        for w in ('baseline','late_v'):
            k=c['compiled'][w]
            k['shared_bytes']=int(re.search(r'\bshared=(\d+)',k['metadata'])[1])
            k['ptx_bytes']=file.with_name(f'partition-{w}.ptx').stat().st_size
        for stage in ('partition','complete'):
            summary={};paired={}
            for w in ('baseline','late_v'):
                groups={u:[r for r in data['rows'] if (r['batch'],r['context'],r['dim'],r['stage'],r['variant'],r['unroll'])==(b,n,d,stage,w,u)] for u in (32,128)}
                assert all(len(g)==5 for g in groups.values())
                vals=[r['device_interval_us'] for r in groups[128]];median=st.median(vals)
                delta=abs(st.median(r['device_interval_us'] for r in groups[32])-median)/median
                summary[w]={'median_us':median,'range_us':[min(vals),max(vals)],'convergence_delta':delta,'converged_5pct':delta<=.05}
                paired[w]={r['repeat']:r['device_interval_us'] for r in groups[128]}
            ratios=[paired['baseline'][i]/paired['late_v'][i] for i in range(5)]
            rows.append({'batch':b,'context':n,'dim':d,'stage':stage,'variants':summary,
                'speedup_baseline_over_late_v':summary['baseline']['median_us']/summary['late_v']['median_us'],
                'paired_ratio_median':st.median(ratios),'paired_ratio_range':[min(ratios),max(ratios)],
                'both_converged':all(r['converged_5pct'] for r in summary.values())})
    assert len(cases)==9 and len(rows)==18
    checks=[c for c in num['checks'] if 'fp32_close' in c]
    comparisons=[c for c in num['checks'] if c.get('comparison')=='baseline_vs_late_v']
    counts={'fp32_and_graph_checks':len(checks),'outputs_bitwise_equal':sum(c['output']['bitwise_equal'] for c in comparisons),
        'partials_bitwise_equal':sum(c['partials']['bitwise_equal'] for c in comparisons),
        'lse_bitwise_equal':sum(c['lse']['bitwise_equal'] for c in comparisons),'pair_checks':len(comparisons),
        'benchmark_native_equal':sum(c['native_equal'] for c in data['checks']),'benchmark_checks':len(data['checks'])}
    result={'rows':rows,'cases':cases,'numerics':counts,'protocol_tests_passed':6,
        'nonconverged':[{'batch':r['batch'],'context':r['context'],'dim':r['dim'],'stage':r['stage'],'variant':w,**s}
            for r in rows for w,s in r['variants'].items() if not s['converged_5pct']],
        'production_changed':False,'model_logits_tested':False,'hardware_counters_obtained':False}
    code_equal=[c['compiled']['baseline']['machine_code']==c['compiled']['late_v']['machine_code'] for c in cases]
    resources_equal=[all(c['compiled']['baseline'][k]==c['compiled']['late_v'][k] for k in ('registers_per_thread','spill_count','shared_bytes')) for c in cases]
    result['machine_code_equal_cases']=sum(code_equal)
    result['resource_equal_cases']=sum(resources_equal)
    ir=json.loads((ROOT/'results/latev-ir/manifest.json').read_text())
    assert len(ir['entries'])==4 and all(not r['missing'] for r in ir['entries'])
    for entry in ir['entries']:
        c=next(c for c in cases if f"b{c['batch']}-l{c['context']}-d{c['dim']}"==entry['case'])
        assert c['compiled'][entry['variant']]['machine_code']==entry['machine_code']
    result['ir_snapshot_kernels']=len(ir['entries'])
    out=ROOT/'docs/late-v';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    write_report(out,result)
    print(json.dumps({'cases':len(cases),'records':len(data['rows']),'numerics':counts,'nonconverged':len(result['nonconverged']),'code_equal':sum(code_equal),'resources_equal':sum(resources_equal)}))


def write_report(out,result):
    rows,cases,counts=result['rows'],result['cases'],result['numerics']
    identical=result['machine_code_equal_cases']==len(cases) and result['resource_equal_cases']==len(cases)
    conclusion=('本次9个形状的最终执行代码.text逐字节一致，编译资源也相同。仅后移V加载的源码改写被编译链消解，没有形成资源改善。' if identical else '部分形状的执行代码发生变化，需逐形状结合资源和配对计时判断；不能统一切换后端。')
    lines=['# V加载位置控制实验','',conclusion,'',
        '只将V的tl.load从QK之前移动到PV计算之前；QK/PV、exp/log、有序分母累加及最终四lane合并完全保留。tile、分区、4-warps设置和merge相同。AST控制测试会移除这一条V赋值再比较函数体，防止混入其他算术改动。','',
        '| D | B / KV | 原完整 μs | 后移完整 μs | 原/后移 | .text相同 | 均收敛 |',
        '|---:|---:|---:|---:|---:|---|---|']
    for c in sorted(cases,key=lambda c:(c['dim'],c['batch'],c['context'])):
        r=next(r for r in rows if (r['dim'],r['batch'],r['context'],r['stage'])==(c['dim'],c['batch'],c['context'],'complete'))
        same=c['compiled']['baseline']['machine_code']==c['compiled']['late_v']['machine_code']
        lines.append(f"| {c['dim']} | {c['batch']} / {c['context']} | {r['variants']['baseline']['median_us']:.3f} | {r['variants']['late_v']['median_us']:.3f} | {r['speedup_baseline_over_late_v']:.3f} | {'是' if same else '否'} | {'是' if r['both_converged'] else '否'} |")
    lines+=['','![Paired device intervals](latency.png)','',
        '完整与分区计时、五轮范围及同轮配对比值均在summary.json。相同机器代码的小差异不应当成新kernel的优化收益；未收敛项须先复查测量条件。','',
        '## 如何比较生成代码','',
        '不能直接比较整个cubin文件哈希：函数名、调试行号和其他metadata都可能不同。本轮分别提取CUDA ELF64中对应函数的.text section，比较完整执行字节；另行比较n_regs、n_spills与launch动态shared。每个形状保留.text SHA256和字节数、PTX与完整编译metadata。','',
        '| D | B / KV | regs 原→后移 | spill字段 原→后移 | shared bytes 原→后移 | .text bytes |',
        '|---:|---:|---:|---:|---:|---:|']
    for c in sorted(cases,key=lambda c:(c['dim'],c['batch'],c['context'])):
        a,b=c['compiled']['baseline'],c['compiled']['late_v']
        lines.append(f"| {c['dim']} | {c['batch']} / {c['context']} | {a['registers_per_thread']}→{b['registers_per_thread']} | {a['spill_count']}→{b['spill_count']} | {a['shared_bytes']}→{b['shared_bytes']} | {a['machine_code']['bytes']} / {b['machine_code']['bytes']} |")
    lines+=['',
        '执行代码一致支持“源码位置变化没有改变本次编译结果”，但不指明是哪一个编译pass进行了重排，也不是对任意版本和形状的保证。编译spill字段不是动态流量，静态资源也不是achieved occupancy。本轮不额外声称取得硬件计数器。','',
        'B1/L8192的D64与D128各保存原/后移两条路径的TTIR、TTGIR、LLVM IR和PTX，共16份编译文本。快照按实际benchmark metadata哈希定位，并保留对应.text指纹和文件SHA256，见results/latev-ir/manifest.json。IR含函数名和调试信息，不能把原始文本差异直接解释成执行差异。','',
        '## 正确性','',
        f"新增6项CPU测试通过，包含控制launch参数、仅V加载位置变化的AST检查，以及ELF执行段提取检查。三种D的预检输出、partials与LSE均逐位一致。边界矩阵{counts['fp32_and_graph_checks']}次FP32/Graph-eager检查通过；{counts['pair_checks']}组对照中，output {counts['outputs_bitwise_equal']}、partials {counts['partials_bitwise_equal']}、LSE {counts['lse_bitwise_equal']}组逐位一致。性能矩阵{counts['benchmark_checks']}次FP32比较中，{counts['benchmark_native_equal']}次输出与原生逐位一致。",'',
        '覆盖D64/128/256、GQA group 1/4/16、随机物理页、-1补齐和NaN尾部，长度0/1/255/256/257/1023/1024/1025，更新Q/K/V、翻转页表、增长到2051和全空Graph行。容差固定atol=rtol=2e-2。未把本候选接入模型，不增加模型logits或吞吐结论。','',
        '## 下一步','',
        '保留正式实现。本次修改没有提供新的机器实现，不应增加一个功能重复的正式后端。下一步需要检查编译IR中的实际布局、流水与活跃值，再设计编译器无法简单消解的控制实验；例如分别验证K/V分块加载与有序分母片段的真实布局需求，仍保持数值协议。不要直接加入人为barrier来强迫源码顺序：它会同时改变同步开销。','',
        '## 计时与复现','',
        f"9形状、360条重复记录，{len(result['nonconverged'])}/36个配置未通过32/128的5%收敛检查。5轮随机交错variant/stage/unroll，每轮20次replay，主指标为Graph128区间除以算子次数。预热、JIT、capture、正确性检查不计时。逐轮检查所选GPU共享；边界检查在另一张GPU运行，不宣称整台机器独占。",'',
        '未锁频，重复使用同一Q/K/V和元数据，区间包含graph节点间隙，不是纯kernel busy time或真实生成延迟。五轮范围仅描述波动。','',
        '```bash','source tools/env.sh','export CUDA_VISIBLE_DEVICES=0',
        '/path/to/venv/bin/python tools/smoke_late_v.py --output results/new-latev-smoke.json',
        '/path/to/venv/bin/python tools/ablate_late_v.py --task correctness --output results/new-latev-correctness',
        '/path/to/venv/bin/python tools/ablate_late_v.py --task benchmark --output results/new-latev-benchmark','```','',
        '报告生成器默认读取results/latev-benchmark-v1与latev-correctness-v1，复测使用新目录或新的源码副本；历史结果保持原样。','']
    (out/'RESULTS.md').write_bytes(('\n'.join(lines)).encode())


if __name__=='__main__':main()
