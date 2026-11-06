"""Build the bounded numerical-compatibility report from raw scalar records."""
import argparse
import json
from pathlib import Path
import re
from _common import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results')
    p.add_argument('--output',type=Path,default=ROOT/'docs/matched-results')
    args=p.parse_args()
    read=lambda name:json.loads((args.results/name).read_text())
    probes=[]
    for name in ('matched-random-v1','matched-random-v2','matched-activations-large-v1','matched-activations-large-v2'):
        rows=read(name+'/summary.json')['rows']
        probes.append({'source':name,'samples':len(rows),
            'predicted_splits_equal_native_auto':sum(r['predicted_fixed_vs_auto']['bitwise_equal'] for r in rows),
            'matched_bitwise_equal':sum(r['matched_vs_auto']['bitwise_equal'] for r in rows),
            'fp32_tolerance_passed':sum(r['matched_vs_fp32']['close_2e-2'] for r in rows),
            'max_abs_vs_auto':max(r['matched_vs_auto']['max_abs'] for r in rows)})
    models=[]
    for name,label,batch,inputs in (
        ('matched-model-small-b8-v2','0.8B',8,512),
        ('matched-model-large-b8-v2','9B',8,512),
        ('matched-model-large-v2','9B',1,2048)):
        summary=read(name+'/summary.json')
        eager=read(name+'/matched.json')
        graph=read(name+'/matched_graph.json')
        free=read(name+'/matched_free_graph.json')
        models.append({'source':name,'model':label,'batch':batch,'input_tokens':inputs,
            'output_tokens':summary['traced_steps'],'plan':read(name+'/matched-plan.json'),
            'eager_equal_logits_steps':sum(r['vs_native_auto']['bitwise_equal'] for r in eager['steps']),
            'graph_equal_logits_steps':sum(r['bitwise_equal'] for r in graph['steps']),
            'eager_argmax_differences':len(eager['argmax_differences']),
            'graph_argmax_differences':len(graph['argmax_differences']),
            'free_graph_tokens_equal':free['matches_native'],
            'same_input_samples':summary['same_input_samples'],
            'all_logits_equal':eager['all_logits_equal_auto'] and graph['all_logits_equal_auto']})
    previous=read('matched-model-large-v1/summary.json')
    lse_before=read('matched-lse-audit-v2/summary.json')
    lse_after=read('matched-lse-audit-fastlog/summary.json')
    test_log=(args.results/'matched-final-tests.log').read_text()
    match=re.search(r'(\d+) passed',test_log)
    if not match or 'failed' in test_log:
        raise ValueError('final full test suite did not pass')
    data={'scope':'Pinned FlashAttention 2.7.4.post1, BF16, A100, fixed batch/capacity; untimed numerical diagnosis',
        'tests_passed':int(match[1]),'probes':probes,'models':models,
        'b1_9b_argmax_differences_under_native_history':{
            k:len(previous['cases'][k]['argmax_differences']) for k in ('tc','matched')},
        'local_lse_audit_before':lse_before,'local_lse_audit_after':lse_after,
        'all_model_cases_passed':all(r['all_logits_equal'] and r['free_graph_tokens_equal'] for r in models)}
    lines=['# Capacity/LSE 对齐实测结果','',
        '本轮为数值诊断，不包含正式性能计时。step 从0计数；每个 batch step 的比较覆盖整个 batch 的完整词表 logits。','',
        '| 模型 | batch | 输入 / 输出 | splits × 分区token | eager一致steps | Graph一致steps | Graph自由生成 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in models:
        lines.append(f"| {r['model']} | {r['batch']} | {r['input_tokens']} / {r['output_tokens']} | {r['plan']['splits']} × {r['plan']['tokens_per_split']} | {r['eager_equal_logits_steps']}/{r['output_tokens']} | {r['graph_equal_logits_steps']}/{r['output_tokens']} | {'一致' if r['free_graph_tokens_equal'] else '分歧'} |")
    lines+=['','| 算子采样 | 样本数 | 静态预测分区与auto一致 | 自写输出逐位一致 | FP32容差通过 | 最大绝对差 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in probes:
        lines.append(f"| {r['source']} | {r['samples']} | {r['predicted_splits_equal_native_auto']} | {r['matched_bitwise_equal']} | {r['fp32_tolerance_passed']} | {r['max_abs_vs_auto']:.8g} |")
    lines+=['',f"全量测试：{data['tests_passed']} passed。",'',
        'v1 已对齐分区和归一化/LSE协议，但保留 Triton `tl.log`；v2 对 local LSE 和 merge 的 log 使用显式近似 lg2×ln2。',
        'B=1、9B、2048/128：固定1024 TC split-K 在原生历史下有5个 argmax 分歧；v1有2个；v2结果见上表。',
        '固定历史的差异次数与自由生成首次分歧是不同统计量。旧版失败记录和原始结果保留。','',
        '这些固定输入上的逐位一致不构成任意输入、模型、GPU、FlashAttention版本的逐位等价保证。',
        '新路径为显式实验API；原先五个公共后端及默认 flash_attn 保留。正式性能数值仍以旧报告中对应后端的测试为准。','',
        '原始数据目录和配置由 summary.json 的 source / plan 字段给出；复现步骤见 ../MATCHED_SPLITKV.md。','']
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    (args.output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'all_model_cases_passed':data['all_model_cases_passed'],'tests_passed':data['tests_passed'],'model_cases':len(models)}))


if __name__=='__main__':
    main()
