"""Report complete paged-operator experiments, with explicit comparison boundaries."""
import argparse,json,statistics
from pathlib import Path
from _common import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,default=ROOT/'results/fpart-benchmark-v1')
    p.add_argument('--correctness',type=Path,default=ROOT/'results/fpart-correctness-v2')
    p.add_argument('--output',type=Path,default=ROOT/'docs/fragment-partition')
    args=p.parse_args();bench=json.loads((args.benchmark/'summary.json').read_text())
    correct=json.loads((args.correctness/'summary.json').read_text())
    assert bench['status']=='completed' and not bench['rejections'] and correct['status']=='passed'
    rows=[];cases=[]
    for d in (64,128):
        for b in (1,8,32):
            for n in (512,2048,8192):
                folder=args.benchmark/f'b{b}-l{n}-d{d}'
                case=json.loads((folder/'case.json').read_text());cases.append(case)
                assert all(case['comparison'][k]['bitwise_equal'] for k in ('output','partials','lse'))
                for stage in ('partition','complete'):
                    variants={}
                    subset=[r for r in bench['rows'] if (r['batch'],r['context'],r['dim'],r['stage'])==(b,n,d,stage)]
                    for variant in ('baseline','shared'):
                        values={u:[r['device_interval_us'] for r in subset if r['variant']==variant and r['unroll']==u] for u in (32,128)}
                        assert all(len(a)==5 for a in values.values())
                        med={u:statistics.median(a) for u,a in values.items()};delta=abs(med[32]-med[128])/med[128]
                        variants[variant]={'median_us':med[128],'range_us':[min(values[128]),max(values[128])],
                            'convergence_delta':delta,'converged_5pct':delta<=.05,'unroll_medians_us':med}
                    ratios=[]
                    for repeat in range(5):
                        pair={v:next(r['device_interval_us'] for r in subset if r['variant']==v and r['unroll']==128 and r['repeat']==repeat) for v in variants}
                        ratios.append(pair['baseline']/pair['shared'])
                    rows.append({'dim':d,'batch':b,'context':n,'stage':stage,'variants':variants,
                                 'speedup':variants['baseline']['median_us']/variants['shared']['median_us'],
                                 'paired_ratios':ratios,'paired_median':statistics.median(ratios)})
    valid=[c for c in correct['checks'] if 'fp32_close' in c]
    pairs=[c for c in correct['checks'] if c.get('comparison')=='baseline_vs_shared']
    production=[c for c in correct['checks'] if c.get('comparison')=='baseline_vs_production']
    native=[c for c in valid if c['native_equal'] is not None]
    assert len(valid)==96 and len(pairs)==48 and len(native)==72
    assert all(c['fp32_close'] and c['graph_eager_equal'] and c['all_empty_zero'] is not False for c in valid)
    assert all(c[k]['bitwise_equal'] for c in pairs for k in ('output','partials','lse'))
    counts={'fp32_graph_outputs':len(valid),'strict_output_partial_lse_pairs':len(pairs),
            'production_baseline_checks':len(production),'native_checks':len(native),'native_equal':sum(c['native_equal'] for c in native),
            'benchmark_fp32_checks':len(bench['checks']),'benchmark_native_equal':sum(c['native_equal'] for c in bench['checks'])}
    not_converged=[{'dim':r['dim'],'batch':r['batch'],'context':r['context'],'stage':r['stage'],'variant':v}
                   for r in rows for v,s in r['variants'].items() if not s['converged_5pct']]
    summary={'rows':rows,'numerics':counts,'nonconverged':not_converged,'resources':[
        {'dim':c['dim'],'batch':c['batch'],'context':c['context'],'compiled':c['compiled']} for c in cases],
        'timing_records':len(bench['rows']),'cpu_tests_passed':9,'production_changed':False,'model_logits_tested':False,
        'scope':'baseline = original matched Triton partition; complete = partition + unchanged merge with preallocated workspace; not a model benchmark'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
    write_report(summary,cases,args.output)
    plot(summary,args.output/'latency.png')
    print(json.dumps({'numerics':counts,'timing_records':len(bench['rows']),'nonconverged':not_converged}))


def write_report(summary,cases,out):
    text='''# 四状态分母原语接入真实分页 partition

已完成独立实验partition接入、严格数值对齐和真实Paged KV延迟矩阵。正式decode配置与模型入口保持原样，D256不属于本实验支持范围。

## 接入与计算顺序

[实验kernel](../../src/paged_decode/experimental_fragment_partition.py)沿用matched_splitkv的分页地址、反向tile遍历、QK、online max/exp、PV、partials/LSE写出及原merge。只替换分母的缩放和更新：将16×4旧状态广播成与P一致的MMA片段，由共享内存原语保序更新，再从首个8-token块的even位置提取四路状态。每个tile只做一次提取归约。最终仍按 `(s0+s2)+(s1+s3)` 合并。

新接口 [prepare](../../tools/fragment_partition_common.py) 在launch前完成编译、MMA-v2 `[1,4]` 布局检查、三个FP32 MMA输入/一个输出检查、128线程和静态＋动态shared预算检查。prepare在Graph捕获之前执行；replay不读取GPU标量决定Python分支。D64/TILE256、D128/TILE128及sm80之外明确报错，没有静默fallback。PROBE仅用于预检，正式计时关闭。

## 接入时发现的FMA边界

第一次MHA扩展检查中，BF16输出及LSE相同，但123个FP32 partial元素出现差异，最大为2.9802322387695312e-8；严格字节门槛阻止了该版本进入计时。将原路径与候选的fp_fusion同时关闭后，差异消失。

进一步对原cubin运行带行号的nvdisasm，发现源码84行的首个even累加在默认模式有4个静态FFMA站点，关闭融合后为0。原先“先把state乘alpha并舍入，再传给微内核”的接口阻断了这次收缩。

修正后的 [partition_state_asm](../../tools/fragment_partition_asm.py) 接收P、未缩放state和alpha。默认模式以显式 `fma.rn.f32` 完成第一个even更新，后续even/odd继续按原顺序使用FP32加法；诊断关闭融合时则显式先乘再加。没有为通过测试修改正式baseline或放宽阈值。修正后原失败样本在两种融合模式的output、partials、LSE均字节一致。

相关证据：[失败对照](../../results/fpart-fusion-diag-v1/summary.json)、[修正对照](../../results/fpart-fusion-diag-v2/summary.json)、[原始行号SASS](../../results/fpart-fusion-diag-v1/baseline-fusionTrue.lines.sass)。这说明Triton源码层面的加法顺序相同，还不足以证明后端生成的舍入序列相同；原微内核使用fp_fusion=False和已舍入初值，不能替代完整partition的验证。

## 正确性与Graph

96次FP32/Graph输出检查通过，48组原／候选的output、normalized FP32 partials、LSE逐字节相同，48次baseline与正式matched_decode对照通过。比较先检查shape/dtype，再转换为uint8比较，因此包含signed zero表示；LSE中的-inf也纳入字节对照。有限输出不得有NaN/Inf；FP32显式gather参考保持atol=rtol=2e-2，并报告最大与均方根误差。

覆盖D64/128、GQA group 1/4/16、默认split规划及显式一个split；后者强制多tile状态连续更新。长度0/1/255/256/257/1023/1024/1025，后续增长至2051；随机物理页，初始无效表项-1及NaN尾部，KV/Q内容更新、页映射翻转和全空行。每次更新后重复Graph replay三次，并与eager的字节表示比较。预检还验证了完整partition中的6144个P片段位置，空上下文CTA没有写probe tags。

非空输入另与固定FlashAttention比较；显式使用相同split数量，避免删除空行后batch变化改变原生启发式。72次原生对照全部字节相同。性能矩阵各形状还重新做FP32与原生输出检查，原／候选输出、partials及LSE一致后才计时。新增9项CPU协议测试在本地、服务器通过。

## 计时边界与矩阵

baseline指本项目原有matched Triton partition，不是FlashAttention性能；原生FlashAttention在本轮用于正确性对照，未纳入计时。partition只运行分区kernel；complete包含partition和原merge，workspace提前分配，包含GPU读写但不包含分配成本。没有模型层、KV写入、token选择或端到端吞吐数据。

batch=1/8/32，KV=512/2048/8192，D=64/128，HQ/HKV=32/8，共18个形状。相同输入、分页布局、split/tile、warp数、merge和workspace几何。fp_fusion=True保持默认配置，实际首项FMA已在候选中显式保留。

5轮独立重复，随机交错variant、stage、32/128调用的Graph；每轮20次replay。主指标为128调用Graph的设备事件区间除以调用数，含节点间隙，不等于kernel busy time。JIT、preflight、预热、正确性、捕获和cuobjdump都在正式区间外。原始计时720条，五轮范围及同轮配对比值见summary.json；不是置信区间。Graph重复使用固定Q/KV，输入和页表更新另做正确性检查，计时不等于连续生成。GPU0逐轮核对宿主PID，记录时钟/功耗/温度；未锁频、不宣称全主机独占。正确性在同型号GPU1完成。

| D | B | KV | partition 原→候选 μs | 比值 | complete 原→候选 μs | 比值 | 两路径/两阶段收敛 |
|---:|---:|---:|---|---:|---|---:|---|
'''
    for d in (64,128):
        for b in (1,8,32):
            for n in (512,2048,8192):
                rs=[r for r in summary['rows'] if (r['dim'],r['batch'],r['context'])==(d,b,n)]
                a=next(r for r in rs if r['stage']=='partition');z=next(r for r in rs if r['stage']=='complete')
                fmt=lambda r:f"{r['variants']['baseline']['median_us']:.3f}→{r['variants']['shared']['median_us']:.3f}"
                conv=all(v['converged_5pct'] for r in rs for v in r['variants'].values())
                text+=f"| {d} | {b} | {n} | {fmt(a)} | {a['speedup']:.3f} | {fmt(z)} | {z['speedup']:.3f} | {'通过' if conv else '未通过'} |\n"
    text+='\n![Complete operator speed ratios](latency.png)\n\n'
    complete=[r for r in summary['rows'] if r['stage']=='complete']
    best=max(complete,key=lambda r:r['paired_median']);worst=min(complete,key=lambda r:r['paired_median'])
    text+=f"同轮配对比值中位数的最高点为D{best['dim']}、B{best['batch']}、KV{best['context']}：{best['paired_median']:.3f}；最低点为D{worst['dim']}、B{worst['batch']}、KV{worst['context']}：{worst['paired_median']:.3f}。\n\n"
    if not any(r['paired_median']<1 for r in complete):
        text+='本矩阵没有观察到候选的配对中位数退化，不人为构造反例。此前[直接块选择实验](../denominator/RESULTS.md)已记录真实退化；本候选仍有共享容量和单warp归约的代价，不能将18个形状外推为任意配置的收益。\n\n'
    text+=f"未通过32/128中位数5%收敛检查的配置数：{len(summary['nonconverged'])}/72。未收敛条目会保留在summary中，不与其他campaign拼接结果。\n\n"
    text+='''## 完整kernel的资源代价

以下取B1/KV8192的实际编译结果。静态shared来自cuobjdump，动态shared来自Triton metadata，两者相加；不能只看metadata忽略inline PTX的静态分配。

| D | 版本 | REG/thread | spill字段 | STACK bytes | LDL / STL静态站点 | shared 静态＋动态 bytes | PTX shuffle站点 |
|---:|---|---:|---:|---:|---|---|---:|
'''
    for d in (64,128):
        case=next(c for c in cases if (c['dim'],c['batch'],c['context'])==(d,1,8192))
        for v in ('baseline','shared'):
            c=case['compiled'][v]
            text+=f"| {d} | {v} | {c['registers_per_thread']} | {c['spill_count']} | {c['stack_bytes']} | {c['static_ldl']} / {c['static_stl']} | {c['static_shared']}＋{c['dynamic_shared']} | {c['ptx_shuffle_sites']} |\n"
    text+='''
候选显式静态缓冲为D64 16640 bytes、D128 8448 bytes。按同型号A100查询的167936 bytes/SM静态容量上限估算，完整D64原路径已受shared限制为最多1 CTA/SM，候选仍为1；完整D128则由2降至1。这与上一轮单CTA微内核的容量背景不同。上界不是achieved occupancy，也没有bank-conflict、带宽或stall计数器；不能仅用REG或PTX文本大小解释全部延迟变化。

所有18个形状的TTIR/TTGIR/LLVM/PTX/cubin、SASS及资源文本随case保留，不能用独立微内核的0 spill结论代替完整kernel数据。新旧.text指纹用于核对实际代码差异；merge实现保持原样。

## 复现与交付范围

```bash
source tools/env.sh
export CUDA_VISIBLE_DEVICES=0
/path/to/venv/bin/python tools/smoke_fragment_partition.py --output results/new-fpart-smoke
/path/to/venv/bin/python tools/ablate_fragment_partition.py --task correctness --output results/new-fpart-correctness
/path/to/venv/bin/python tools/ablate_fragment_partition.py --task benchmark --output results/new-fpart-benchmark
/path/to/venv/bin/python -m pytest tests/test_fragment_partition_protocol.py -q
```

使用固定的sm80/Triton3.2/FlashAttention2.7.4环境。每个输出目录独立，argument/environment记录源码与设备；prepare必须在Graph捕获之前。原始证据索引和旧源码快照保留首次数值失败及两轮融合诊断，失败版本没有计入正式计时。

本轮没有注册新引擎后端，也没有将这些比值写成FlashAttention、真实模型或端到端提速。Qwen3.5当前模型回归所需的D256尚不在该原语的支持范围；进一步进入模型实验前，需要扩展相应tile/片段、重做资源与数值验证，再选择模型入口开关。原模型路径继续使用此前已验证实现。
'''
    (out/'RESULTS.md').write_bytes(text.encode())
    (ROOT/'FRAGMENT_PARTITION.md').write_bytes('''# 保序四状态原语的真实分页partition接入

已完成：[完整实验报告](docs/fragment-partition/RESULTS.md)、[数值与性能汇总](docs/fragment-partition/summary.json)。

- 支持sm80、4 warps、D64/TILE256与D128/TILE128，保持原split规划和merge。
- 修复了原始alpha缩放与首次加法的FMA收缩差异，默认baseline配置保持原样。
- 分页边界、GQA、空行、KV/页表更新及Graph验证通过；比较包含严格字节表示。
- 18个真实Paged KV形状分别测partition和partition＋merge；数值与原生FlashAttention对照，性能与原matched Triton实现对照。
- 正式模型入口未改变；D256和模型性能是后续独立验收，不能外推本轮算子比值。
'''.encode())


def plot(summary,path):
    import os,math
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    rows=[r for r in summary['rows'] if r['stage']=='complete']
    lim=max(1,max(abs(math.log2(r['speedup'])) for r in rows))
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,d in zip(axes,(64,128)):
        values=[[next(r for r in rows if (r['dim'],r['batch'],r['context'])==(d,b,n)) for n in (512,2048,8192)] for b in (1,8,32)]
        img=ax.imshow([[math.log2(r['speedup']) for r in a] for a in values],cmap='BrBG',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
        for i,row in enumerate(values):
            for j,r in enumerate(row):
                a=r['variants']['baseline']['median_us'];z=r['variants']['shared']['median_us']
                ax.text(j,i,f"{r['speedup']:.2f}x\n{a:.1f} us\n-> {z:.1f} us",ha='center',va='center',fontsize=9,
                        color='white' if abs(math.log2(r['speedup']))>lim*.65 else '#162b2b')
        ax.set_xticks(range(3),['512','2048','8192']);ax.set_yticks(range(3),['1','8','32'])
        ax.set_xlabel('KV context');ax.set_ylabel('Batch');ax.set_title(f'D{d}: partition + merge')
    fig.suptitle('Original matched Triton / shared-state candidate (median Graph interval)',fontsize=12)
    fig.subplots_adjust(left=.07,right=.88,wspace=.3,top=.84,bottom=.12)
    cax=fig.add_axes([.91,.18,.02,.6]);fig.colorbar(img,cax=cax,label='log2 ratio; positive = candidate faster')
    fig.savefig(path,dpi=180);plt.close(fig)


if __name__=='__main__':main()
