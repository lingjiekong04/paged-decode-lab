"""Aggregate paired model runs; reject missing/duplicate repeats or GPU sharing."""
import json,math,statistics
from _common import ROOT


def aggregate(rows):
    result=[]
    for n in (512,2048):
        for mode in ('eager','graph'):
            values={}
            for backend in ('native','original','shared'):
                group=[r for r in rows if (r['input_tokens'],r['mode'],r['backend'])==(n,mode,backend)]
                if len(group)!=5 or sorted(r['repeat'] for r in group)!=list(range(5)):
                    raise ValueError('five unique rounds required for each configuration')
                if not all(r['tokens_equal_native'] and math.isfinite(r['wall_seconds']) and r['wall_seconds']>0 and len(r['decode_event_ms'])==63
                           and all(math.isfinite(x) and x>0 for x in r['decode_event_ms']) for r in group):
                    raise ValueError('invalid timing or correctness record')
                wall=[r['wall_seconds'] for r in group];device=[r['decode_event_median_ms'] for r in group]
                values[backend]={'wall_median_s':statistics.median(wall),'wall_range_s':[min(wall),max(wall)],
                    'tps_median':statistics.median(r['output_tokens_per_second'] for r in group),
                    'decode_median_ms':statistics.median(device),'decode_range_ms':[min(device),max(device)],
                    'peak_allocated_range':[min(r['peak_allocated_bytes'] for r in group),max(r['peak_allocated_bytes'] for r in group)]}
            ratios={}
            for base in ('native','original'):
                a={r['repeat']:r for r in rows if (r['input_tokens'],r['mode'],r['backend'])==(n,mode,base)}
                b={r['repeat']:r for r in rows if (r['input_tokens'],r['mode'],r['backend'])==(n,mode,'shared')}
                ratios[base]={'wall_ratios':[a[i]['wall_seconds']/b[i]['wall_seconds'] for i in range(5)],
                              'decode_ratios':[a[i]['decode_event_median_ms']/b[i]['decode_event_median_ms'] for i in range(5)]}
                ratios[base].update(wall_paired_median=statistics.median(ratios[base]['wall_ratios']),
                                    decode_paired_median=statistics.median(ratios[base]['decode_ratios']))
            result.append({'input_tokens':n,'mode':mode,'variants':values,'ratios':ratios})
    return result


def main():
    output=ROOT/'docs/fragment-model-performance';output.mkdir(parents=True,exist_ok=True)
    models=[]
    for label,phase in [('0.8B','mperf-small-v1'),('9B','mperf-large-v1')]:
        folder=ROOT/'results'/phase;data=json.loads((folder/'summary.json').read_text())
        monitor=json.loads((folder/'gpu-monitor.json').read_text())
        assert data['status']=='passed' and not monitor['violations'] and monitor['rows']
        assert len(data['rows'])==60 and len(data['validation'])==12
        assert all(s['bytes_equal'] for r in data['validation'] for s in r['steps'])
        models.append({'model':label,'revision':data['model']['revision'],'rows':aggregate(data['rows']),
                       'gpu_monitor_samples':len(monitor['rows']),'validation_logit_steps':sum(len(r['steps']) for r in data['validation']),
                       'callback_invocations':data['callback_invocations']})
    summary={'models':models,'timed_records':120,'wall_runs':120,'event_runs':120,'selected_gpu_exclusive':True,
             'host_exclusive':False,'production_changed':False,'scope':'B1 fixed synthetic inputs; official Torch linear-attention fallback; retained buffers identical'}
    (output/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
    report='''# 空闲GPU上的完整模型性能对照

固定Qwen3.5-0.8B/9B revision，三条路径各测eager/Graph：native为原生FlashAttention，original为原matched Triton，shared为新四状态partition。native及原matched均在本轮重新计时，不跨campaign拼接比值。正式默认后端不变。

## 测量协议

- 每模型batch1、输入512/2048个固定seed的合成token IDs，输出64 token、忽略EOS提前停止。相同模型内两种长度均使用容量2112、完整页表和相同KV池；prompt IDs保留在inputs.json。不是自然请求长度分布或服务负载。
- 在计时前，对六种配置、两种长度重跑64步完整logits字节检查及free greedy对照；callback计数确认实验路由被使用。prefill保持原生FlashAttention，线性注意力是固定官方Torch fallback。
- 5轮随机交错12个配置。每次切换时清理旧Graph对象，重新捕获、预热均在区间外。模型加载、JIT、正确性与GPU查询不计入正式区间。
- 主wall测量不插入逐步CUDA事件：包含prefill、完整decode、token选择、最终同步及文本后处理。TPS=64/wall，不能称为纯decode吞吐。
- 另起独立生成调用测63个decode forward事件区间，包含输入缓冲/位置更新及执行间隙，排除prefill和token选择；每轮取这些区间的中位数，再报告五轮中位数。这不是kernel busy time或仅attention耗时。
- 120条记录分别来自120次wall生成与120次事件生成。每次生成均检查与原生free greedy token序列相同。五轮范围为描述性波动，不是置信区间。

GPU0测0.8B，GPU1测9B，同机并行；每张卡启动前无compute进程，启动后建立独占host PID集合，每秒监控并在每次正式调用前后核验。新增进程或监测失败会使campaign无效。未锁频，时钟、功耗、温度和全部监测记录保留；不宣称主机CPU或整机独占。

两个matched路径使用同样的query copy、每层资源布局及原merge，只替换partition；native也保留同一批预分配实验workspace。因此峰值显存是本实验固定预算下的实际allocated peak，不是各后端最小显存。Graph池随配置重建，不能用workspace字段代替完整显存。

## 五轮结果

比值=基线路径/新路径，>1表示新路径更快；配对比值按同轮计算。所有绝对值为五轮中位数。

| 模型 | 输入 | 模式 | wall秒 原生/原matched/新 | TPS 原生/原matched/新 | decode事件ms 原生/原matched/新 | wall配对 原matched/新 | wall配对 原生/新 |
|---|---:|---|---|---|---|---:|---:|
'''
    for model in models:
        for row in model['rows']:
            vals=row['variants'];fmt=lambda key,prec:'/'.join(f'{vals[v][key]:.{prec}f}' for v in ('native','original','shared'))
            report+=f"| {model['model']} | {row['input_tokens']} | {row['mode']} | {fmt('wall_median_s',3)} | {fmt('tps_median',2)} | {fmt('decode_median_ms',3)} | {row['ratios']['original']['wall_paired_median']:.3f} | {row['ratios']['native']['wall_paired_median']:.3f} |\n"
    graph_ratios=[r['ratios']['original']['wall_paired_median'] for m in models for r in m['rows'] if r['mode']=='graph']
    report+=f"\nGraph模式下原matched/新路径的同轮wall比值中位数范围为{min(graph_ratios):.4f}–{max(graph_ratios):.4f}。结合五轮重叠范围，本组数据没有显示稳定明显的模型端到端收益。eager波动更大，不以单次最好值下结论。\n"
    report+='\n## 峰值allocated显存\n\n以下为五轮最大值（GiB），含统一保留的workspace，非最小后端预算。\n\n| 模型 | 输入 | 模式 | 原生 | 原matched | 新路径 |\n|---|---:|---|---:|---:|---:|\n'
    for model in models:
        for row in model['rows']:
            vals=[row['variants'][v]['peak_allocated_range'][1]/(1024**3) for v in ('native','original','shared')]
            report+=f"| {model['model']} | {row['input_tokens']} | {row['mode']} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |\n"
    report+='''
![Generation wall time](wall.png)

逐轮原始wall、全部decode事件、五轮范围、配对比值及峰值显存见 [summary.json](summary.json) 和各r*.json；不能把微内核或partition比值写成模型提速。预填充、Gated DeltaNet、投影/MLP、lm_head和调度间隙共同决定端到端时间；attention占比与D256原语在真实形状下的收益仍需分段测量。接近1的比值应结合五轮波动解读，本轮没有硬件计数器分解，不能据此精确归因。

## 复现

```bash
source tools/env.sh
export CUDA_VISIBLE_DEVICES=0
/path/to/venv/bin/python tools/benchmark_fragment_models.py --model models/Qwen3.5-0.8B --output results/new-model-performance
# 9B以同样参数单独运行；所选卡必须空闲。
python tools/report_fragment_model_performance.py
```

benchmark参数、设备UUID、源码SHA和模型身份见每轮arguments/environment.json，GPU独占证据见gpu-monitor.json。此接口只支持本轮固定batch与模型范围；不是请求级P99、在线调度、并行通信或工业服务压测。
'''
    (output/'RESULTS.md').write_bytes(report.encode())
    plot(models,output/'wall.png')
    (ROOT/'MODEL_PERFORMANCE.md').write_bytes('''# 实验partition的完整模型性能

[报告](docs/fragment-model-performance/RESULTS.md) · [汇总](docs/fragment-model-performance/summary.json)

原生FlashAttention、原matched与新四状态路径，在0.8B/9B的eager/Graph上重新对照。生成wall与decode事件独立测量，固定预算、五轮配对，并核验所选GPU无其他计算进程。正式默认入口未改动。
'''.encode())
    print(json.dumps({'timed_records':120,'models':models}))


def plot(models,path):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,model in enumerate(models):
        for j,n in enumerate((512,2048)):
            ax=axes[i,j]
            for k,(mode,backend) in enumerate((m,b) for m in ('eager','graph') for b in ('native','original','shared')):
                row=next(r for r in model['rows'] if r['input_tokens']==n and r['mode']==mode)
                v=row['variants'][backend];mid=v['wall_median_s'];lo,hi=v['wall_range_s']
                ax.barh(k,mid,color={'native':'#718096','original':'#507bab','shared':'#188176'}[backend],alpha=.8)
                ax.errorbar(mid,k,xerr=[[mid-lo],[hi-mid]],fmt='none',color='#162b2b',capsize=3)
            ax.set_yticks(range(6),['Native eager','Original eager','Shared eager','Native Graph','Original Graph','Shared Graph'])
            ax.invert_yaxis();ax.set_title(f"{model['model']} / {n} input tokens");ax.set_xlabel('Generation wall seconds (64 output tokens)')
            ax.set_axisbelow(True);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Fixed model runs: median and five-round range; separate GPUs')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


if __name__=='__main__':main()
