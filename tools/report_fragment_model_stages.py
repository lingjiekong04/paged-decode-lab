"""Validate and summarize the diagnostic D256 model-stage campaign."""
import argparse
import json
import math
from pathlib import Path
import statistics


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('timing must be finite and positive')
    return value


def describe(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'raw': values}


def aggregate(data, repeats=5, steps=16):
    if data.get('status') != 'passed':
        raise ValueError('campaign did not complete')
    metadata = data['metadata']
    layers = {n: sorted(m['layer'] for m in metadata if m['input_tokens'] == n) for n in (512, 2048)}
    if not layers[512] or layers[512] != layers[2048]:
        raise ValueError('snapshot layer set incomplete')
    for n in layers:
        if len(set(layers[n])) != len(layers[n]):
            raise ValueError('duplicate snapshots')
    if any(not m['all_outputs_bytes_equal_native'] or m['context'] != m['input_tokens']+steps for m in metadata):
        raise ValueError('real-tensor numerical gate failed')
    records = {}
    expected = {(n,b,flag,r) for n in (512,2048) for b in ('native','original','shared')
                for flag in (False,True) for r in range(repeats)}
    for row in data['rows']:
        key = (row['input_tokens'], row['backend'], row['instrumented'], row['repeat'])
        if key not in expected or key in records:
            raise ValueError('unexpected or duplicate model observation')
        samples = row['steps']
        if len(samples) != steps or [s['step'] for s in samples] != list(range(1,steps+1)):
            raise ValueError('incomplete decode window')
        for s in samples:
            positive(s['outer_ms'])
            if not s['logits_bytes_equal'] or s['context'] != row['input_tokens']+s['step']:
                raise ValueError('decode validation failed')
            stage = s['stages_ms']
            if row['instrumented']:
                required = {'forward'} | {f'{kind}.{i}' for i in layers[key[0]] for kind in ('module','operator')}
                if set(stage) != required:
                    raise ValueError('stage ranges incomplete')
                for v in stage.values():
                    positive(v)
                for i in layers[key[0]]:
                    if stage[f'operator.{i}'] > stage[f'module.{i}']:
                        raise ValueError('nested operator exceeds module range')
                total = sum(stage[f'module.{i}'] for i in layers[key[0]])
                if total > stage['forward'] or stage['forward'] > s['outer_ms']:
                    raise ValueError('invalid interval nesting')
            elif stage is not None:
                raise ValueError('plain graph unexpectedly instrumented')
        records[key] = row
    if set(records) != expected:
        raise ValueError('missing model observations')
    observations = {}
    wanted = {(n,i,b,s,r) for n in (512,2048) for i in layers[n] for b in ('native','original','shared')
              for s in (('full',) if b == 'native' else ('full','partition','merge')) for r in range(repeats)}
    for row in data['micro']:
        key = tuple(row[k] for k in ('input_tokens','layer','backend','stage','repeat'))
        if key not in wanted or key in observations:
            raise ValueError('unexpected or duplicate micro observation')
        positive(row['us_per_call'])
        if row['context'] != row['input_tokens']+steps or min(row['unroll'],row['replays']) < 1:
            raise ValueError('invalid micro protocol')
        observations[key] = row['us_per_call']
    if set(observations) != wanted:
        raise ValueError('incomplete micro matrix')
    result = []
    for n in (512,2048):
        variants = {}
        for b in ('native','original','shared'):
            plain, tagged, forwards, ops, modules, shares, overheads = [],[],[],[],[],[],[]
            for r in range(repeats):
                p = statistics.median(s['outer_ms'] for s in records[n,b,False,r]['steps'])
                tagged_samples = records[n,b,True,r]['steps']
                t = statistics.median(s['outer_ms'] for s in tagged_samples)
                f = [s['stages_ms']['forward'] for s in tagged_samples]
                op = [sum(s['stages_ms'][f'operator.{i}'] for i in layers[n]) for s in tagged_samples]
                mod = [sum(s['stages_ms'][f'module.{i}'] for i in layers[n]) for s in tagged_samples]
                plain.append(p); tagged.append(t); forwards.append(statistics.median(f))
                ops.append(statistics.median(op)); modules.append(statistics.median(mod))
                shares.append(statistics.median(a/v for a,v in zip(op,f)))
                overheads.append(t/p-1)
            stages = {}
            for stage in (('full',) if b == 'native' else ('full','partition','merge')):
                by_layer = {str(i): describe([observations[n,i,b,stage,r] for r in range(repeats)]) for i in layers[n]}
                stages[stage] = {'layers': by_layer,
                    'sum_layer_medians_us': sum(v['median'] for v in by_layer.values())}
            variants[b] = {'plain_outer_ms': describe(plain), 'tagged_outer_ms': describe(tagged),
                'tagged_forward_ms': describe(forwards), 'tagged_operator_sum_ms': describe(ops),
                'tagged_full_module_sum_ms': describe(modules), 'tagged_operator_fraction': describe(shares),
                'tagging_overhead_fraction': describe(overheads), 'isolated': stages}
        ratios = {stage: describe([sum(observations[n,i,'original',stage,r] for i in layers[n]) /
                                    sum(observations[n,i,'shared',stage,r] for i in layers[n])
                                    for r in range(repeats)]) for stage in ('full','partition','merge')}
        result.append({'input_tokens': n, 'snapshot_context': n+steps, 'full_layers': layers[n],
                       'variants': variants, 'isolated_original_over_shared': ratios})
    return result


def plot(models, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    labels=[];core=[];rest=[]
    for m in models:
        name=m['model']['repo_id'].split('-')[-1]
        group=next(r for r in m['results'] if r['input_tokens']==2048)
        for backend in ('original','shared'):
            v=group['variants'][backend]
            labels.append(name+' '+backend)
            core.append(v['tagged_operator_sum_ms']['median'])
            rest.append(v['tagged_forward_ms']['median']-core[-1])
    ys=list(range(len(labels)))
    a.barh(ys,core,color='#147d92',label='Attention callback')
    a.barh(ys,rest,left=core,color='#c4cbd6',label='Other forward intervals')
    a.set_yticks(ys,labels);a.invert_yaxis();a.set_xlabel('Instrumented forward interval (ms)')
    a.set_title('Input 2048: diagnostic Graph ranges');a.legend(loc='lower right',fontsize=8)
    names=[];ratios=[];lows=[];highs=[];colors=[]
    for m in models:
        name=m['model']['repo_id'].split('-')[-1]
        for r in m['results']:
            for stage,color in [('partition','#147d92'),('full','#b46b22')]:
                v=r['isolated_original_over_shared'][stage]
                names.append(f"{name} / {r['snapshot_context']} / {stage}")
                ratios.append(v['median']);lows.append(v['median']-v['min']);highs.append(v['max']-v['median']);colors.append(color)
    xs=list(range(len(names)))
    b.barh(xs,ratios,color=colors,height=.65)
    b.errorbar(ratios,xs,xerr=[lows,highs],fmt='none',color='#253446',capsize=3)
    b.axvline(1,color='#5c6570',linestyle='--',linewidth=1)
    b.set_yticks(xs,names,fontsize=8);b.invert_yaxis()
    b.set_xlabel('Original / shared latency (>1: shared faster)')
    b.set_title('Isolated real tensors: five-round min–max')
    fig.suptitle('D256 model-stage diagnosis | batch 1 | A100 | not primary wall timing',fontsize=12)
    fig.savefig(output/'stages.png',dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--small',type=Path,required=True)
    p.add_argument('--large',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    models=[]
    for path in (args.small,args.large):
        data=json.loads((path/'summary.json').read_text())
        monitor=json.loads((path/'gpu-monitor.json').read_text())
        if monitor['violations'] or not monitor['rows']:
            raise ValueError('GPU monitoring failed')
        config=json.loads((path/'arguments.json').read_text())
        models.append({'model':data['model'],'gpu_uuid':config['gpu_uuid'],
                       'description':config['description'],'plan':config['plan'],
                       'monitor_samples':len(monitor['rows']),'results':aggregate(data)})
    summary={'models':models,'production_changed':False,'scope':'Diagnostic external-event Graph ranges; isolated hot-cache real-tensor microbenchmarks. Not primary wall timing.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(models,args.output)
    custom=[v['tagged_operator_fraction']['median'] for m in models for r in m['results']
            for b,v in r['variants'].items() if b!='native']
    overhead=[v['tagging_overhead_fraction']['median'] for m in models for r in m['results'] for v in r['variants'].values()]
    small=models[0]['results']
    large=models[1]['results']
    long_large=next(r for r in large if r['input_tokens']==2048)
    p_ratio=long_large['isolated_original_over_shared']['partition']['median']
    f_ratio=long_large['isolated_original_over_shared']['full']['median']
    benefit=[r['isolated_original_over_shared']['full']['median'] for r in small+[large[0]]]
    lines=['# D256 真实模型分段分析','',
      '固定 revision 的 Qwen3.5-0.8B／9B，batch=1，容量 2112，输入 512／2048；每个配置独立重复 5 轮。',
      '每轮按固定 native greedy 前缀前进 16 个真实 decode 步，逐步校验完整 logits 的字节一致性。',
      '所有计时都在 Graph 模式下完成；此次为诊断实验，不替代前一轮独立 wall 吞吐实验。','',
      '## 结论','',
      f"自定义 attention 回调占带插桩 forward 的约 {min(custom):.2%}–{max(custom):.2%}。整体插桩开销约 {min(overhead):.2%}–{max(overhead):.2%}，因此占比只作诊断，不能视作无扰动的精确占比。其余 forward 区间占主要时间；尚未把这些区间进一步归因于线性 attention、MLP 或 lm_head。",'',
      f"真实张量的独立 hot-cache 测量中，0.8B 两种长度和 9B 短上下文的完整算子比值为 {min(benefit):.4f}–{max(benefit):.4f}（original/shared）。但 9B、上下文 2064 的 partition 比值为 {p_ratio:.4f}，即 shared 延迟增加约 {(1/p_ratio-1):.2%}；完整算子延迟增加约 {(1/f_ratio-1):.2%}。优化收益依赖实际模型形状与上下文，不能全面替换基线。",'',
      '这一结果支持上一轮端到端收益很小的解释：被优化区间本身较小，且一个实际长上下文形状存在退化。独立 hot-cache 延迟与模型内执行不同，不能直接换算成生产吞吐收益。当前线性 attention 使用官方 Torch fallback，此结论限定于本项目当前执行路径。','',
      '下一项 kernel 工作优先定位 9B 的 16 query heads／4 KV heads、D256、上下文 2064 退化场景，对比两版 partition 的 shared memory、寄存器与实际并发；目前未取得硬件计数器证据，不能预先判定根因。若目标转为整体吞吐，再细分其余 forward 区间。','',
      '![分段分析](stages.png)','', '## 模型内分段与插桩开销','',
      '利用 external CUDA event 节点记录完整 forward、full-attention 模块及其内部 attention 回调。',
      '模块范围包含投影、位置编码、KV 写入、attention 及输出处理；回调范围包含算子及自定义后端的 Q 拷贝。',
      '每步完成后同步并在计时外读取 logits；plain 与 tagged 路径使用同一同步协议。',
      '下表为每轮 16 步中位数，再取 5 轮中位数；占比在每步内计算。插桩开销按同轮 tagged/plain−1 计算。',
      '事件范围包含内部 GPU 调度间隙，不能视作纯 kernel busy time；嵌套范围不能重复相加。','',
      '| 模型 | 输入 | 后端 | plain step ms | tagged forward ms | attention 回调总 ms | full-attn 模块总 ms | 回调/forward | 插桩开销 |',
      '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for m in models:
        name=m['model']['repo_id'].split('/')[-1]
        for r in m['results']:
            for b,v in r['variants'].items():
                med=lambda k:v[k]['median']
                lines.append(f"| {name} | {r['input_tokens']} | {b} | {med('plain_outer_ms'):.4f} | {med('tagged_forward_ms'):.4f} | {med('tagged_operator_sum_ms'):.4f} | {med('tagged_full_module_sum_ms'):.4f} | {med('tagged_operator_fraction'):.2%} | {med('tagging_overhead_fraction'):+.2%} |")
    lines += ['', '插桩会改变执行时间与调度；若开销较大，占比仅支持粗略定位，不能用它计算精确的生产加速上限。',
      '本轮实例 GPU UUID 与上一轮不同，绝对延迟不与旧服务器数字直接配对；所有对照均来自本轮同卡实验。','',
      '## 真实 Q/K/V 的独立算子测量','',
      '从 native 第 16 个 decode 步提取每个 full-attention 层的真实 Q/K/V，实际上下文为 528／2064。',
      '保留容量 2112 和实际分页表；snapshot Q 为克隆后的紧凑张量；独立计时不包含 snapshot／Q 拷贝／workspace 分配。',
      '每个 stage 单独捕获 64 次调用，再 replay 20 次，5 轮随机顺序。所有完整输出先通过 native 字节一致性校验。',
      '重复访问同一层数据，属于 hot-cache 诊断；层间实际缓存竞争、launch 间隙和数据相关性不同，不能把独立阶段相加当模型延迟。',
      '表中延迟为各层 5 轮中位数之和；比值为每轮各层延迟求和后 original/shared 的 5 轮中位数，>1 表示 shared 较快。','',
      '| 模型 | 上下文 | stage | original µs | shared µs | original/shared | 五轮范围 |',
      '|---|---:|---|---:|---:|---:|---|']
    for m in models:
        name=m['model']['repo_id'].split('/')[-1]
        for r in m['results']:
            for stage,ratio in r['isolated_original_over_shared'].items():
                a=r['variants']['original']['isolated'][stage]['sum_layer_medians_us']
                b=r['variants']['shared']['isolated'][stage]['sum_layer_medians_us']
                lines.append(f"| {name} | {r['snapshot_context']} | {stage} | {a:.3f} | {b:.3f} | {ratio['median']:.4f} | {ratio['min']:.4f}–{ratio['max']:.4f} |")
    lines += ['', '完整原始值、逐层结果、各轮范围及配置见 `summary.json` 和对应 results/stage-*-v1。',
      '## 复现','', '```bash',
      'CUDA_VISIBLE_DEVICES=0 python tools/profile_fragment_model_stages.py --model models/Qwen3.5-0.8B --output results/stage-small-v1',
      'CUDA_VISIBLE_DEVICES=1 python tools/profile_fragment_model_stages.py --model models/Qwen3.5-9B --output results/stage-large-v1',
      'python tools/report_fragment_model_stages.py --small results/stage-small-v1 --large results/stage-large-v1 --output docs/fragment-model-stages',
      '```','', '默认输出目录必须不存在；GPU 必须空闲，发生外部进程占用会使实验失败。两模型可各用一张卡，CPU 与主机仍共享。',
      '生产默认入口保持不变；实验独立注册回调并记录源文件快照。线性 attention 沿用现有官方 Transformers Torch fallback。','',
      'External event 接口依据：[CUDA 12.4 Runtime API](https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_Runtime_API.pdf)。', '']
    (args.output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'models':len(models),'groups':sum(len(m['results']) for m in models)}))


if __name__=='__main__':
    main()
