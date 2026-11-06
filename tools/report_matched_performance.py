"""Summarize the isolated five-repeat matched split-KV performance campaign."""
import json,statistics as st
from pathlib import Path
from _common import ROOT


def main():
    results=ROOT/'results'
    read=lambda p:json.loads(p.read_text())
    ops=[]; models=[]
    for model in ('small','large'):
        folder=results/f'mperf-op-{model}'
        rows=read(folder/'summary.json')['rows']
        keys=sorted({(r['batch'],r['context'],r['backend']) for r in rows})
        for batch,context,backend in keys:
            selected=[r for r in rows if (r['batch'],r['context'],r['backend'])==(batch,context,backend)]
            assert len(selected)==5
            prof=read(folder/f'profile-b{batch}-l{context}-d256-{backend}.json')
            busy=sum(e['duration_us'] for e in prof['cuda_events'])/prof['iterations']
            ops.append({'model':model,'batch':batch,'context':context,'backend':backend,
                'graph_us':st.median(r['graph_event_us'] for r in selected),
                'graph_us_range':[min(r['graph_event_us'] for r in selected),max(r['graph_event_us'] for r in selected)],
                'preallocated_eager_event_us':st.median(r['preallocated_event_us'] for r in selected),
                'full_allocation_wall_us':st.median(r['full_call_summary_us']['median'] for r in selected),
                'profile_kernel_busy_us':busy,'profile_kernel_names':sorted({e['name'] for e in prof['cuda_events']}),
                'workspace_bytes':selected[0].get('matched_workspace_bytes',0)+selected[0]['workspace_bytes'],
                'max_abs_vs_fp32':max(r['errors']['max_abs'] for r in selected),
                'plan':selected[0]['matched_plan']})
        for batch in (1,8):
            data=read(results/f'mperf-model-{model}-b{batch}'/'summary.json')
            for backend in ('flash_attn','triton_tc_splitk','matched_splitkv'):
                for mode in ('eager','graph'):
                    selected=[r for r in data['rows'] if (r['backend'],r['mode'])==(backend,mode)]
                    assert len(selected)==5
                    val=next(r for r in data['validation'] if (r['backend'],r['mode'])==(backend,mode))
                    tps=[r['output_tokens_per_second'] for r in selected]
                    models.append({'model':model,'batch':batch,'backend':backend,'mode':mode,
                        'throughput':st.median(tps),'throughput_range':[min(tps),max(tps)],
                        'decode_step_ms':st.median(r['decode_model_ms']['median'] for r in selected),
                        'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in selected),
                        'workspace_bytes':selected[0]['workspace_bytes'],'plan':selected[0]['plan'],
                        'bitwise_equal_logits_steps':sum(r['bitwise_equal'] for r in val['steps']),
                        'free_greedy_matching_repeats':sum(r['free_tokens_match_native'] for r in selected)})
    out=ROOT/'docs/matched-performance'
    out.mkdir(parents=True,exist_ok=True)
    data={'operator_repeat_rows':270,'model_repeat_rows':120,'operator_summary':ops,'model_summary':models,
        'scope':'Single A100 GPU0, serial jobs; Torch threads4; pinned models/dependencies; prefill native and linear attention Torch fallback.'}
    (out/'summary.json').write_bytes((json.dumps(data,indent=2)+'\n').encode())
    lines=['# Matched split-KV 性能实测','',
        '两档真实Qwen3.5几何，各9个算子形状（B=1/8/32，KV=512/2048/8192，D256），三条路径各重复5次，共270条记录。四组模型负载、六配置各重复5次，共120次生成。全部任务串行运行于GPU0，GPU1保持空闲；torch CPU线程数固定4。','',
        '路径：原生FlashAttention 2.7.4.post1、固定1024分区的triton_tc_splitk、容量/LSE对齐的matched_splitkv。后者仍为实验API。','',
        '## 算子','',
        '下表均为5次Graph CUDA-event区间的中位数（微秒），包括kernel间执行间隙；不是纯kernel busy time。完整矩阵、eager区间、含资源分配的同步调用wall time、独立profiler busy time、workspace与波动范围见summary.json。原生API自己管理输出和workspace。','',
        '| 模型几何 | batch / KV | 原生 μs | 固定1024 μs | 对齐 μs | 对齐/原生耗时 |',
        '|---|---:|---:|---:|---:|---:|']
    for model in ('small','large'):
        for b,n in ((1,512),(1,8192),(8,2048),(32,8192)):
            v={r['backend']:r['graph_us'] for r in ops if (r['model'],r['batch'],r['context'])==(model,b,n)}
            lines.append(f"| {'0.8B' if model=='small' else '9B'} | {b} / {n} | {v['flash_attn']:.2f} | {v['triton_tc_splitk']:.2f} | {v['matched_splitkv']:.2f} | {v['matched_splitkv']/v['flash_attn']:.3f} |")
    lines+=['','## 完整模型','',
        'B1输入2048、B8输入512，输出128，三条路径使用同一prompt IDs、KV容量和同一官方模型。所有配置在计时前单独对照原生历史的完整logits，并预热；正式运行采用自由greedy生成。吞吐wall time包括prefill和token选择，decode step单独计时包括KV写入、linear state和lm_head。','',
        '| 模型 | batch | 模式 | 原生 tok/s | 固定1024 tok/s | 对齐 tok/s | 对齐/原生吞吐 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for model in ('small','large'):
        for b in (1,8):
            for mode in ('eager','graph'):
                v={r['backend']:r['throughput'] for r in models if (r['model'],r['batch'],r['mode'])==(model,b,mode)}
                lines.append(f"| {'0.8B' if model=='small' else '9B'} | {b} | {mode} | {v['flash_attn']:.2f} | {v['triton_tc_splitk']:.2f} | {v['matched_splitkv']:.2f} | {v['matched_splitkv']/v['flash_attn']:.3f} |")
    matched=[r for r in models if r['backend']=='matched_splitkv']
    lines+=['',f"对齐路径：8组配置共{sum(r['bitwise_equal_logits_steps'] for r in matched)}/1024个batch steps的logits与原生逐位一致；40次正式自由生成中{sum(r['free_greedy_matching_repeats'] for r in matched)}/40次token序列一致。固定1024的数值分歧保留在原始验证记录中，其性能应结合不同greedy轨迹理解。",'',
        '不能把数值对齐直接解释成提速。更细的分区增加并行度，也增加partial/LSE写回、空分区调度、exp/log及归约成本；两阶段实现还多一次launch。完整模型中attention仅占部分开销，linear attention、MLP、权重读取和prefill都限制端到端收益。表中退化结果同样保留。','',
        '本轮未锁定GPU时钟，五次重复按配置连续运行，未随机交错后端次序。范围明显重叠时，不能仅凭几个百分点的中位数差异认定稳定提升。波动范围见summary.json与图中误差条。','',
        '![Measured performance](performance.png)','','## 复现','',
        '```bash','source tools/env.sh','export CUDA_VISIBLE_DEVICES=0',
        '.venv35/bin/python tools/benchmark_matched_operator.py --qwen35 --profile --output results/new-op-small',
        '# 换 --qwen35-large 测9B几何；默认9形状、5重复、50次迭代。',
        '.venv35/bin/python tools/benchmark_matched_model.py --model models/Qwen3.5-9B --output results/new-model-b1',
        '# B8添加 --batch 8 --input-tokens 512；另一档换0.8B模型。',
        '.venv35/bin/python tools/report_matched_performance.py',
        'python tools/plot_matched_performance.py  # requires matplotlib report extra','```','',
        '原始记录为results/mperf-*；环境、工具SHA256、模型revision、逐次结果与验证均保留。报告不使用Nsight硬件计数器，profiler为独立运行的CUDA kernel duration样本。','']
    (out/'RESULTS.md').write_bytes(('\n'.join(lines)).encode())
    print(json.dumps({'operators':len(ops),'model_configs':len(models),'matched_logits_equal_steps':sum(r['bitwise_equal_logits_steps'] for r in matched)}))


if __name__=='__main__':main()
