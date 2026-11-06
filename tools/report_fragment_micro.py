"""Report only retained micro-harness measurements; no engine speedup claim."""
import json,statistics
from _common import ROOT


def main():
    raw=ROOT/'results/fragment-ablation-v2'
    data=json.loads((raw/'summary.json').read_text())
    recurrence=json.loads((ROOT/'results/fragment-recurrence-v1/summary.json').read_text())
    assert data['status']==recurrence['status']=='passed'
    assert len(data['checks'])==len(recurrence['checks'])==112
    assert all(c['denominator_cpu_bitwise'] and c['graph_eager_equal'] and c['finite'] for c in data['checks'])
    assert all(c['four_states_cpu_bitwise'] and c['graph_eager_equal'] and c['finite'] for c in recurrence['checks'])
    out=ROOT/'docs/fragment-micro';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in (64,128):
        variants={}
        for v in ('baseline','shared'):
            values={u:[r['device_interval_us'] for r in data['timings'] if r['d']==d and r['variant']==v and r['unroll']==u] for u in (32,128)}
            assert all(len(s)==5 for s in values.values())
            med={u:statistics.median(s) for u,s in values.items()}
            delta=abs(med[32]-med[128])/med[128]
            variants[v]={'median_us':med[128],'range_us':[min(values[128]),max(values[128])],
                'unroll_medians_us':med,'convergence_relative_delta':delta,'converged_5pct':delta<=.05}
        pairs=[]
        for repeat in range(5):
            pair={v:next(r['device_interval_us'] for r in data['timings'] if r['d']==d and r['variant']==v and r['unroll']==128 and r['repeat']==repeat) for v in variants}
            pairs.append(pair['baseline']/pair['shared'])
        rows.append({'d':d,'variants':variants,'ratio_of_medians':variants['baseline']['median_us']/variants['shared']['median_us'],
                     'paired_ratios':pairs,'paired_ratio_median':statistics.median(pairs)})
    summary={'rows':rows,'resources':data['resources'],'timing_records':len(data['timings']),
             'row_sum_checks':112,'four_state_recurrence_checks':112,'paired_cases_per_harness':56,
             'cpu_tests_passed':9,'production_changed':False,'model_logits_tested':False,
             'scope':'Single-CTA QK/probabilities/ordered denominator/PV harness; no paged KV, normalization, split merge or engine timing'}
    (out/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
    text='''# MMA 片段映射与保序分母微内核

本轮完成了实际线程映射验证、两种保序共享内存微内核，以及正确性／Graph／编译资源／计时实验。候选值得进入完整partition接入实验，尚未采用为正式后端。

## 1. 探针验证的布局

固定A100 sm80、Triton 3.2、4 warps，QK的MMA-v2布局为 `warpsPerCTA=[1,4]`。通过 `inline_asm_elementwise` 将每个P元素标记为 `thread_id * pack + pack_slot`，再按逻辑位置写出；D64/TILE256的4096个位置、D128/TILE128的2048个位置都与下面映射一致，且没有重复或遗漏。

设 `w=thread//32`、`l=thread%32`、`slot=4*j+i`：

```text
row    = l//4 + 8*(i//2)
column = 8*w + 32*j + 2*(l%4) + i%2
pack   = TILE/8               # 32 or 16 FP32 elements per thread
```

这里的slot是本实验inline-asm参数包的槽位，不是SASS物理寄存器编号；Triton接口没有承诺跨版本保持pack次序。官方PTX文档给出单个m16n8k16累加器片段的lane分布，本实验另外验证了Triton跨warp及重复tile的展开次序。[NVIDIA PTX片段布局](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816)

例如thread0的前四个槽是 `(row,col)=(0,0),(0,1),(8,0),(8,1)`；接下来的四个槽将column增加32。thread32属于warp1，起始column为8。四个warp分别持有block编号模4为0/1/2/3的数据，不能只在原warp内相加就得到按block全序累加的结果。

完整映射CSV：[D64](../../results/fragment-ablation-v2/d64-t256/ownership.csv)、[D128](../../results/fragment-ablation-v2/d128-t128/ownership.csv)。每次实验都重新运行映射探针；候选在launch前检查sm80、MMA版本、warp分布，以及inline asm的两个输入和输出仍为同一MMA布局。结构不符会报错。

## 2. 候选如何保持数值顺序

用Triton调用显式inline PTX；不引入CUDA C++构建链。128个线程按已验证的片段映射，把整个FP32 P暂存到共享内存，一次CTA同步后由warp0读取完整block序列。warp0的每个lane处理两行、每行四路累加中的一路，依次执行 `block0 even→odd，block1 even→odd，…` 的FP32加法。

提供两个输出形式：

- `shared_ordered_asm(tile)` 合并为 `(s0+s2)+(s1+s3)`，供本轮计时harness使用；显式shuffle取得pair0/pair1，避免lane相关的最终合并顺序。
- `shared_ordered_asm(tile, states=True)` 在合并前返回四路状态，供连续tile更新验证和后续partition接入使用。不能将前一种总和重新分配为四路状态。

第二次CTA同步确保结果写好后，所有warp才读取它。每次调用覆盖全部P和结果缓冲；两个barrier均由128个线程参与，warp0内部没有部分lane的shuffle掩码。共享地址进入算术前显式转为地址寄存器，inline asm标记side effect及memory clobber。

```mermaid
flowchart LR
  A[128 threads: MMA P fragments] --> B[Stage FP32 P once in shared]
  B --> C[CTA barrier]
  C --> D[warp0: ordered updates of four states]
  D --> E[Return four states or ordered row sum]
  E --> F[CTA barrier]
  F --> G[All warps read results]
```

实现：[片段与PTX生成器](../../tools/fragment_asm.py)、[独立GPU harness](../../src/paged_decode/experimental_fragment_micro.py)。这是针对所列布局的实验原语，不能仅根据逻辑shape移植到其他GPU、warp数、MMA版本或D256。

## 3. 正确性和连续状态验证

行和harness共112项检查（56组原／候选对照），分母与CPU显式逐块FP32参考逐位一致，PV anchor与原路径逐位一致，Graph/eager一致，输出有限。覆盖两个几何、四种初始状态（0、随机、跨数量级、subnormal）、长度0/1/7/8/9/TILE-1/TILE，以及K/V的NaN尾部；每次输入更新后连续replay四次。

四状态harness另外完成112项检查、56组对照：每组经过7个不同长度的tile更新，并在更新之间对四路状态施加相同的非负alpha缩放。每一步的全部16×4状态与CPU逐块参考逐位一致，原／候选P及PV也逐位相同；Graph重复replay与eager一致。alpha为测试生成值，不等于已经验证完整online-softmax的max/LSE状态机。空长度在这里表示加零的微内核输入，不替代正式引擎的空行处理验收。

新增9项CPU测试分别在本地和远端通过。本报告的“逐位一致”使用torch.equal比较有限FP32数值，未另外区分+0/-0的位编码；没有宣称NaN payload一致。

原始记录：[行和检查](../../results/fragment-ablation-v2/summary.json)、[四状态检查](../../results/fragment-recurrence-v1/summary.json)、[远端CPU测试](../../results/fragment-cpu-remote-tests.log)。没有新增真实模型logits、greedy生成或分页attention的正确性声明。

## 4. 最终代码与资源代价

下面是**行和计时harness**的编译结果；四状态harness另有原始资源记录，未将它的延迟混入本表。

| D | 版本 | REG/thread | spill字段 | STACK bytes | 静态LDL / STL | PTX shuffle站点 | 静态shared | 动态shared |
|---:|---|---:|---:|---:|---|---:|---:|---:|
'''
    for r in data['resources']:
        text+=f"| {r['d']} | {r['variant']} | {r['registers']} | {r['spills']} | {r['stack_bytes']} | {r['static_ldl']} / {r['static_stl']} | {r['ptx_shuffle_sites']} | {r['static_shared']} | {r['dynamic_shared']} |\n"
    text+='''
两种几何的最终 `.text` 均发生变化；D64代码段从229376降至29312 bytes，D128从61184降至23936 bytes。SASS及cuobjdump资源输出保留在对应case目录中。这些是静态指令站点和资源字段，不是动态spill次数或硬件计数器。

代价是额外的共享内存、两次CTA同步，以及归约阶段只使用一个warp。D64的总shared从67584变为84032 bytes；按此前同型号A100查询到的每SM 167936 bytes估算，单看shared容量，并驻CTA上界由2变为1。D128从69632变为77888 bytes，按同一容量上界仍为2。此处没有测achieved occupancy；当前grid只有一个CTA，也没有据此声称大grid一定退化。D128的REG还从168增加到176，不能概括成“所有维度寄存器都更少”。

当前P共享暂存为行主序，可能存在bank冲突；本轮没有取得bank-conflict计数器，不给出动态冲突倍数。后续可以单独比较block/parity/row/lane存储顺序，但不能与引擎接入同时改动后再归因。

## 5. 计时范围与结果

每次harness调用包括QK、row-max/exp、分母，以及维持真实MMA布局与P用途的未归一化PV anchor；只有一个CTA。两条路径固定输入、dtype、tile、warp数与fp_fusion=False。**不是纯分母耗时，不含分页KV寻址、online-softmax多tile循环、split merge、输出归一化或模型执行。**

5轮随机交错variant/unroll，Graph分别捕获32/128次调用，每轮20次replay；主指标为Graph128的设备事件区间除以调用次数。JIT、预热、资源检查、正确性与Graph捕获不在计时区间内；共40条原始重复记录。使用GPU0并逐轮核对宿主PID，不宣称整台主机独占；未锁频，GPU状态快照随timing.json保留。

| D | 原harness μs中位数 [五轮范围] | 候选 μs中位数 [五轮范围] | 中位数比值 原/候选 | 同轮比值中位数 | 32/128收敛 |
|---:|---|---|---:|---:|---|
'''
    for r in rows:
        a,b=r['variants']['baseline'],r['variants']['shared']
        fmt=lambda v:f"{v['median_us']:.3f} [{v['range_us'][0]:.3f}, {v['range_us'][1]:.3f}]"
        text+=f"| {r['d']} | {fmt(a)} | {fmt(b)} | {r['ratio_of_medians']:.3f} | {r['paired_ratio_median']:.3f} | {'通过' if a['converged_5pct'] and b['converged_5pct'] else '未通过'} |\n"
    text+='''
![Five independent repeats and medians](latency.png)

五轮范围是描述性波动，不是置信区间；D128波动较明显，应结合逐轮配对记录解读。Graph事件区间包含节点间隙，不等于kernel busy time。D64在此harness的收益不能填写成完整算子、模型吞吐或端到端提速。

## 6. 复现、失败记录与下一步

```bash
source tools/env.sh
export CUDA_VISIBLE_DEVICES=0
/path/to/venv/bin/python tools/probe_fragment_micro.py --output results/new-fragment-smoke
/path/to/venv/bin/python tools/ablate_fragment_micro.py --output results/new-fragment-ablation
/path/to/venv/bin/python tools/check_fragment_recurrence.py --output results/new-fragment-recurrence
/path/to/venv/bin/python -m pytest tests/test_fragment_asm.py -q
```

固定环境与源码哈希见每轮environment/arguments.json。首次PTX预编译因共享符号直接参与地址加法失败，改为先mov到地址寄存器；首次性能脚本因Triton 3.2懒初始化尚无n_regs字段失败，改为在布局检查后加载函数句柄再取资源。两项均发生在正式计时前，错误日志与当时源码快照完整保留，没有当作通过结果。

行和正式实验为fragment-ablation-v2，四状态实验为fragment-recurrence-v1。补充states输出之后，已检查默认生成的inline PTX字符串/constraints/pack逐字相同、计时harness函数AST不变；正式实验仍以当时source snapshot哈希为准。[补充审计](../../results/fragment-final-source-check.json)及[原始证据索引](../../results/FRAGMENT_EVIDENCE_INDEX.json)保留可核验的对应关系。

下一步只把**四状态版本**接入实验partition：处理寄存器中的旧state与alpha，继续保持block及四lane顺序，重新审计完整kernel的布局和静态＋动态shared预算，再运行分页边界、GQA、Graph更新及原生对齐测试。通过后才测真实partition/complete调用和模型；本轮不修改正式decode后端，也不预设完整算子能够复现微内核比值。
'''
    (out/'RESULTS.md').write_bytes(text.encode())
    (ROOT/'FRAGMENT_MICROKERNEL.md').write_bytes('''# MMA片段与保序分母微内核

已完成：[实验报告](docs/fragment-micro/RESULTS.md)、[机器可读汇总](docs/fragment-micro/summary.json)。

- GPU探针验证6144个逻辑位置的thread/pack-slot映射，无重复遗漏。
- 行和微内核112项、四状态连续更新112项GPU检查通过；新增9项CPU测试在本地及远端通过。
- 实现共享暂存＋保序累加，提供合并行和、四路状态两种输出。正式后端未改变。
- 性能数字仅属于单CTA QK/softmax/denominator/PV测试harness；不代表分页算子或模型收益。

下一步：将四状态版本接入实验partition，再验证真实布局、共享预算、数值和性能。
'''.encode())
    plot(rows,data['timings'],out/'latency.png')
    print(json.dumps({'rows':rows,'row_sum_checks':112,'recurrence_checks':112,'cpu_tests':9}))


def plot(rows,records,path):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    colors={'baseline':'#687889','shared':'#137c72'}
    for ax,row in zip(axes,rows):
        for i,v in enumerate(('baseline','shared')):
            s=row['variants'][v];median=s['median_us'];lo,hi=s['range_us']
            ax.barh(i,median,color=colors[v],height=.44,alpha=.85)
            ax.errorbar(median,i,xerr=[[median-lo],[hi-median]],color='#202c38',capsize=4,fmt='none')
            samples=[r['device_interval_us'] for r in records if r['d']==row['d'] and r['variant']==v and r['unroll']==128]
            ax.scatter(samples,[i+.1]*5,s=12,color='#202c38',zorder=5)
            ax.text(hi+.02*max(s['range_us'][1] for s in row['variants'].values()),i,f'{median:.2f}',va='center',fontsize=9)
        ax.set_yticks([0,1],['Baseline','Shared ordered']);ax.invert_yaxis()
        ax.set_title(f"D{row['d']}: single-CTA harness");ax.set_xlabel('Graph device interval / call (us)')
        ax.set_xlim(0,max(s['range_us'][1] for s in row['variants'].values())*1.22)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('QK + probabilities + ordered denominator + PV; 5 repeats',fontsize=12)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


if __name__=='__main__':main()
