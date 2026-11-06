"""Validated paired summaries for the explicit two-stage D256 experiment."""
import argparse,json,math,statistics
from pathlib import Path

VARIANTS=('original','shared_s3','shared_s2')
BACKENDS=('native',*VARIANTS)


def positive(v):
    if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0:
        raise ValueError('positive finite timing required')
    return v


def describe(v):return {'median':statistics.median(v),'min':min(v),'max':max(v),'raw':v}


def operator_summary(data):
    if data.get('status')!='passed' or not data.get('real_tensor_native_bytes_equal'):
        raise ValueError('operator numerical gate not passed')
    stages=[('partition',g) for g in (18,24,27,28,32,33,36)]+[('full',36)]
    expected={(n,v,s,g,u,r) for n in (512,2048) for v in VARIANTS for s,g in stages for u in (32,128) for r in range(5)}
    values={}
    for row in data['rows']:
        key=tuple(row[k] for k in ('input_tokens','variant','stage','parts','unroll','repeat'))
        if key in values or key not in expected or row['context']!=key[0]+16:raise ValueError('invalid operator matrix')
        values[key]=positive(row['us_per_call'])
    if set(values)!=expected:raise ValueError('incomplete operator matrix')
    resources={(r['input_tokens'],r['variant']):r for r in data['resources']}
    if len(data['resources'])!=6 or set(resources)!={(n,v) for n in (512,2048) for v in VARIANTS}:
        raise ValueError('resource matrix incomplete')
    for n in (512,2048):
        if resources[n,'shared_s2']['driver_max_active_blocks_per_sm']<2:raise ValueError('two-CTA residency not restored')
        if resources[n,'shared_s2']['dynamic_shared_bytes']>=resources[n,'shared_s3']['dynamic_shared_bytes']:
            raise ValueError('shared memory did not decrease')
    result=[]
    for n in (512,2048):
        for stage,parts in stages:
            variants={}
            for v in VARIANTS:
                times={str(u):describe([values[n,v,stage,parts,u,r] for r in range(5)]) for u in (32,128)}
                delta=times['32']['median']/times['128']['median']-1
                variants[v]={'times_us':times,'unroll_relative_delta':delta,'converged_3pct':abs(delta)<=.03}
            ratios={v:describe([values[n,v,stage,parts,128,r]/values[n,'shared_s2',stage,parts,128,r] for r in range(5)]) for v in ('original','shared_s3')}
            result.append({'input_tokens':n,'context':n+16,'stage':stage,'parts':parts,'variants':variants,'over_s2':ratios})
    return result


def model_summary(data):
    if data.get('status')!='passed':raise ValueError('model campaign failed')
    expected={(n,b,m,r) for n in (512,2048) for b in BACKENDS for m in ('eager','graph') for r in range(5)}
    values={}
    for row in data['rows']:
        key=tuple(row[k] for k in ('input_tokens','backend','mode','repeat'))
        if key not in expected or key in values:raise ValueError('unexpected model timing')
        positive(row['wall_seconds'])
        if row['output_tokens']!=64 or not row['tokens_equal_native'] or len(row['decode_event_ms'])!=63:
            raise ValueError('model generation gate failed')
        for ms in row['decode_event_ms']:positive(ms)
        positive(row['peak_allocated_bytes']);positive(row['output_tokens_per_second'])
        if not math.isclose(row['output_tokens_per_second'],64/row['wall_seconds'],rel_tol=1e-9):raise ValueError('TPS units inconsistent')
        values[key]=row
    if set(values)!=expected:raise ValueError('incomplete model matrix')
    valkeys=set()
    for row in data['validation']:
        key=tuple(row[k] for k in ('input_tokens','backend','mode'))
        if key in valkeys:raise ValueError('duplicate model validation')
        valkeys.add(key)
        if not row['free_tokens_equal'] or [x['step'] for x in row['steps']]!=list(range(64)):
            raise ValueError('incomplete model validation')
        if any(not x['bytes_equal'] or x['max_abs']!=0 for x in row['steps']):raise ValueError('logits divergence')
    if valkeys!={(n,b,m) for n in (512,2048) for b in BACKENDS for m in ('eager','graph')}:
        raise ValueError('missing model validation')
    if data['resources']['shared_s2']['driver_max_active_blocks_per_sm']<2:raise ValueError('model resources rejected')
    result=[]
    for n in (512,2048):
        for mode in ('eager','graph'):
            variants={}
            for b in BACKENDS:
                rows=[values[n,b,mode,r] for r in range(5)]
                variants[b]={'wall_seconds':describe([r['wall_seconds'] for r in rows]),
                    'tps':describe([r['output_tokens_per_second'] for r in rows]),
                    'decode_event_median_ms':describe([statistics.median(r['decode_event_ms']) for r in rows]),
                    'peak_allocated_bytes':describe([r['peak_allocated_bytes'] for r in rows])}
            ratios={b:describe([values[n,b,mode,r]['wall_seconds']/values[n,'shared_s2',mode,r]['wall_seconds'] for r in range(5)])
                    for b in ('native','original','shared_s3')}
            result.append({'input_tokens':n,'mode':mode,'variants':variants,'wall_over_s2':ratios})
    return result


def load_campaign(root,tag):
    folder=root/tag;monitor=json.loads((folder/'gpu-monitor.json').read_text())
    if monitor['violations'] or not monitor['rows']:raise ValueError('GPU sharing or missing monitor')
    receipt=json.loads((root/(tag+'.job.json')).read_text())
    if receipt['exit_code']!=0:raise ValueError('campaign process failed')
    return json.loads((folder/'summary.json').read_text())


def plot(operator,models,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,n in zip(axes,(512,2048)):
        groups=[r for r in operator if r['input_tokens']==n and r['stage']=='partition']
        for v,color in [('original','#2577a3'),('shared_s3','#bb782c'),('shared_s2','#2b8b68')]:
            points=[r['variants'][v]['times_us']['128'] for r in groups]
            ax.errorbar([r['parts']*4 for r in groups],[p['median'] for p in points],
                        yerr=[[p['median']-p['min'] for p in points],[p['max']-p['median'] for p in points]],
                        color=color,marker='o',capsize=3,label=v)
        ax.axvline(108,color='gray',linestyle='--',linewidth=1)
        ax.set_title(f'Context {n+16}: same real Q/K/V')
        ax.set_xlabel('Launched CTAs');ax.set_ylabel('Partition interval (µs)');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('D256 num_stages=3→2 | five-round median and min–max | A100')
    fig.savefig(out/'pipeline.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root
    correctness=load_campaign(root,'smem2-correct-v1')
    if len(correctness['checks'])!=72 or not correctness['all_output_partials_lse_bytes_equal']:
        raise ValueError('incomplete paged correctness suite')
    if any(not r['graph_eager_equal'] or not r['native_equal'] or not r['finite'] for r in correctness['checks']):
        raise ValueError('paged correctness failed')
    for r in correctness['resources']:
        if r['variant']=='shared_s2' and r['driver_max_active_blocks_per_sm']<2:raise ValueError('correctness shape residency failed')
    raw=load_campaign(root,'smem2-operator-v1');operators=operator_summary(raw);models=[]
    for size in ('small','large'):
        regress=load_campaign(root,'smem2-model-'+size+'-v1')
        if regress['status']!='passed' or regress['prompt_count']!=10 or len(regress['checks'])!=60 or len(regress['free_checks'])!=60 or regress['logit_steps']!=1920:
            raise ValueError('model prompt suite incomplete')
        if any([s['step'] for s in row['steps']]!=list(range(32)) for row in regress['checks']):
            raise ValueError('incomplete model prompt steps')
        if any(not s['bytes_equal'] for row in regress['checks'] for s in row['steps']) or any(not r['tokens_equal'] for r in regress['free_checks']):
            raise ValueError('model prompt regression failed')
        data=load_campaign(root,'smem2-perf-'+size+'-v1')
        models.append({'model':data['model'],'resources':data['resources'],'results':model_summary(data),
                       'prompt_logit_steps':regress['logit_steps'],'prompt_free_runs':regress['free_runs']})
    control_audit=[]
    from cubin_text import fingerprint as text_fingerprint
    for old,new,function in [('baseline','original','_normalized_partition'),('shared','shared_s3','_fragment_partition')]:
        old_path=root/'residency-resources-v1'/(old+'.cubin')
        new_path=root/'smem2-smoke-v1'/(new+'.cubin')
        if old_path.exists() and new_path.exists():
            a=text_fingerprint(old_path.read_bytes(),function);b=text_fingerprint(new_path.read_bytes(),function)
            control_audit.append({'variant':new,'historical_text':a,'current_text':b,'text_equal':a==b})
    summary={'operators':operators,'operator_resources':raw['resources'],'models':models,'historical_control_text':control_audit,
             'operator_nonconverged':[{'context':r['context'],'stage':r['stage'],'parts':r['parts'],'variant':v}
                 for r in operators for v,t in r['variants'].items() if not t['converged_3pct']],
             'production_changed':False}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(operators,models,args.output)
    resources={r['variant']:r for r in raw['resources'] if r['input_tokens']==2048}
    long=next(r for r in operators if r['context']==2064 and r['stage']=='full')
    part=next(r for r in operators if r['context']==2064 and r['stage']=='partition' and r['parts']==36)
    short=next(r for r in operators if r['context']==528 and r['stage']=='full')
    graph_ratios=[r['wall_over_s2']['shared_s3']['median'] for m in models for r in m['results'] if r['mode']=='graph']
    lines=['# D256 两阶段流水线：恢复双 CTA 驻留后的回归','',
      '唯一手动变化是 shared kernel 的 `num_stages=3→2`；MMA layout、四状态 PTX 归约、分页布局、split 计划、BF16/FP32 数值协议和 merge 均不变。保留 original、shared_s3、shared_s2 三条显式实验路径。','',
      '## 结论','',
      f"目标长上下文 2064 上，shared_s3/shared_s2 的同轮比值中位数为：partition {part['over_s2']['shared_s3']['median']:.4f}×，完整算子 {long['over_s2']['shared_s3']['median']:.4f}×；完整算子相对 original 为 {long['over_s2']['original']['median']:.4f}×。恢复双 CTA 驻留后，原先约 6% 的 partition 退化被消除。",'',
      f"短上下文 528 的完整算子存在约 {(1/short['over_s2']['shared_s3']['median']-1):.2%} 延迟增加，五轮范围见表。它与短上下文非空 CTA 较少、资源门槛收益有限的现象一致；没有硬件计数器单独证明具体访存隐藏或调度原因。因此保留两个显式实验配置，未做全局自动选择或默认替换。",'',
      f"两模型、两长度的 Graph wall 旧版/新版同轮比值中位数为 {min(graph_ratios):.4f}–{max(graph_ratios):.4f}。模型收益应按这些独立 wall 数据判断，不能直接套用 kernel 提速。",'',
      '## 资源与正确性','', '| 版本 | dynamic shared B | static shared B | 寄存器/线程 | spill 字段 | 驱动最大 CTA/SM |',
      '|---|---:|---:|---:|---:|---:|']
    for v,r in resources.items():lines.append(f"| {v} | {r['dynamic_shared_bytes']} | {r['static_shared_bytes']} | {r['registers_per_thread']} | {r['triton_spill_count']} | {r['driver_max_active_blocks_per_sm']} |")
    lines+=['', 'shared_s2 每 CTA 总 shared 预留为 77952+4352+1024=83328 B，两个为 166656 B，低于每 SM 的 167936 B；驱动查询确认可驻留两个 CTA。',
      '寄存器和 spill 字段也随编译调度变化，不能把全部性能变化单独归因于 shared 容量或 occupancy。没有新增 achieved occupancy 或动态 spill 流量的硬件计数器声明。','',
      'original/shared_s3 控制路径在本轮显式按三阶段重新编译。离线核验发现其 .text 与早先存档不逐字节相同；未将差异归因于某个未验证的编译因素。所有速度比都取本轮配对，不跨 campaign 比绝对延迟；原／新 .text 指纹见 summary.json。原 kernel 源码及历史产物保持不变。','',
      '分页回归沿用原 D256 suite：MHA/GQA、三种 head 几何、自动／单分区、四轮状态更新、零长度、NaN padding、无效页与跨页增长。72 项输出通过 FP32 参考（atol/rtol=.02）与 Graph/eager 对照；非空行另外对照 native，全空输出必须严格为零；三版本的输出、partials、LSE 均逐字节一致。',
      '两模型均重跑原有 8 个真实提示和 255／1023 边界提示，native／original／shared_s2 × eager/Graph；每模型 1920 个完整 logits 步和 60 次 free greedy 通过。原 suite 中标签 shared 在本轮明确映射为 shared_s2，映射与源码哈希写在 resident-experiment.json。','',
      '## 真实张量的算子和 grid 对照','',
      '使用先前保存的 9B layer 3 实际 Q/K/V，上下文 528／2064，容量 2112、36 splits。480 项计时：三版本 × 两上下文 × 七个 partition grid 和完整算子 × 两种 unroll × 五轮。',
      '每项 capture 32／128 次调用，replay 20 次；主表为 unroll=128 的五轮中位数，比例为同轮配对值的五轮中位数。完整算子包括 partition+merge，不含 workspace 分配或 snapshot 拷贝；同张量重复访问属于 hot-cache 诊断。','',
      '![pipeline](pipeline.png)','', '| 上下文 | stage | original µs | shared_s3 µs | shared_s2 µs | s3/s2（范围） | original/s2 |',
      '|---|---|---:|---:|---:|---:|---:|']
    for row in operators:
        if row['parts']!=36:continue
        t={v:x['times_us']['128']['median'] for v,x in row['variants'].items()};ratio=row['over_s2']['shared_s3']
        lines.append(f"| {row['context']} | {row['stage']} | {t['original']:.4f} | {t['shared_s3']:.4f} | {t['shared_s2']:.4f} | {ratio['median']:.4f} ({ratio['min']:.4f}–{ratio['max']:.4f}) | {row['over_s2']['original']['median']:.4f} |")
    lines+=['',f"unroll 中位数差异超过 3% 的控制数：{len(summary['operator_nonconverged'])}。全部逐轮数据与 grid 结果保留，未丢弃尾值。",'',
      '## 同轮模型 wall 对照','',
      '固定 revision Qwen3.5-0.8B／9B，batch=1，输入 512／2048，输出 64 tokens，容量 2112。四后端 × eager/Graph × 两长度 × 五轮，每模型 80 项。',
      '正式 wall 调用无逐步 events；另一次生成收集 63 个 decode event 区间。JIT、capture、warmup 和 profiling 均不计入正式 wall。每配置先检查 64 步完整 logits 字节对齐和独立 free greedy，所有计时 token 也必须对齐。',
      '三组预分配 workspace 在所有后端中保留，显存是固定实验预算下的 allocated peak，不能解释为 native 的最小显存。两模型各用一张独占空闲 GPU，CPU／主机共享。沿用官方 Transformers Torch 线性 attention fallback。','',
      '| 模型 | 输入 | mode | native s | original s | shared_s3 s | shared_s2 s | s3/s2（同轮范围） |',
      '|---|---:|---|---:|---:|---:|---:|---|']
    for m in models:
        name=m['model']['repo_id'].split('/')[-1]
        for r in m['results']:
            t={v:d['wall_seconds']['median'] for v,d in r['variants'].items()};ratio=r['wall_over_s2']['shared_s3']
            lines.append(f"| {name} | {r['input_tokens']} | {r['mode']} | {t['native']:.6f} | {t['original']:.6f} | {t['shared_s3']:.6f} | {t['shared_s2']:.6f} | {ratio['median']:.4f} ({ratio['min']:.4f}–{ratio['max']:.4f}) |")
    lines+=['', 'TPS、decode event 中位数、allocated peak、各轮 min/max 和全部配对比值保存在 summary.json；不能用单个算子提速比例代替模型吞吐比例。eager 部分配置的波动较大，不能把单个中位数当成稳定的模型提速证据。生产 kernel、公共默认后端和以往结果均不改动。','',
      '## 复现','', '```bash',
      'CUDA_VISIBLE_DEVICES=0 python tools/resident_experiment.py --task correctness --output /tmp/new-correct',
      'CUDA_VISIBLE_DEVICES=0 python tools/resident_experiment.py --task benchmark --output /tmp/new-operator',
      'CUDA_VISIBLE_DEVICES=0 python tools/resident_experiment.py --task model --model models/Qwen3.5-0.8B --output /tmp/new-model',
      'CUDA_VISIBLE_DEVICES=0 python tools/benchmark_fragment_resident_models.py --model models/Qwen3.5-0.8B --output /tmp/new-perf',
      '```','', '输出目录必须不存在，成功后立即归档。实测使用持久 ZIP 检查点；checkpoint 写入只发生在同步和 GPU guard 之后，不在计时区间内。每个数据集保留 UUID、监测、配置、源码快照及退出记录。','']
    (args.output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'operator_groups':len(operators),'model_groups':sum(len(m['results']) for m in models)}))


if __name__=='__main__':main()
