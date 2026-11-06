"""Verify evidence from the public CLI, without injecting attention implementations."""
import json,re
from _common import ROOT


def main():
    read=lambda p:json.loads((ROOT/p).read_text())
    rows=[]
    for model in ('small','large'):
        reference=read(f'results/mperf-model-{model}-b8/bench-flash_attn-eager-r0.json')
        native=reference['token_ids']
        identity=read(f'results/mperf-model-{model}-b8/arguments.json')['model_identity']
        for mode in ('eager','graph'):
            val=read(f'results/entry-{model}-{mode}-validate/summary.json')
            runtime=read(f'results/entry-{model}-{mode}-validate/runtime.json')
            assert val['status']=='passed' and val['bitwise_equal_steps']==val['tested_batch_steps']==256
            assert runtime['backend']=='triton_matched' and runtime['splitkv_plan'] is not None
            b8=read(f'results/entry-{model}-{mode}-b8/summary.json')
            configured=read(f'results/entry-{model}-{mode}-b8/arguments.json')['model']
            assert all(configured[k]==identity[k] for k in ('repo_id','revision','weights','files'))
            live=read(f'results/entry-{model}-{mode}-b8/runtime.json')
            assert b8['status']=='completed' and len(b8['rows'])==1
            ids=b8['rows'][0]['token_ids']
            assert b8['rows'][0]['prompt_sha256']==reference['prompt_sha256']
            assert ids==native,(model,mode,'greedy tokens differ from native')
            assert live['batch']==8 and live['backend']=='triton_matched' and live['max_context']==1024
            assert live['mode']==mode and live['pages_per_full_layer']==8*reference['kv_table_width']
            rows.append({'model':model,'mode':mode,'short_prompt_logit_steps':256,
                'bitwise_equal_steps':val['bitwise_equal_steps'],'batch8_generation_tokens':sum(map(len,ids)),
                'batch8_matches_native':True,'batch8_plan':live['splitkv_plan'],
                'workspace_bytes':live['workspace_bytes']})
    log=(ROOT/'results/entry-final-tests.log').read_text()
    match=re.search(r'(\d+) passed',log)
    assert match and 'failed' not in log
    out=ROOT/'docs/formal-entry';out.mkdir(parents=True,exist_ok=True)
    data={'tests_passed':int(match[1]),'cli_configs':rows,'kernel_changed':False,
        'logits_steps_bitwise_equal':sum(r['bitwise_equal_steps'] for r in rows),
        'batch8_native_matching_tokens':sum(r['batch8_generation_tokens'] for r in rows)}
    (out/'summary.json').write_bytes((json.dumps(data,indent=2)+'\n').encode())
    lines=['# 正式后端接入验收','',
        '新增公开选项triton_matched，通过Qwen3.5 CLI和原有HybridRunner attention分支直接调用已验证kernel。资源分配、元数据约束和完整decode Graph由runner负责，无临时AttentionInterface注入或原生decode回退。','',
        '| 模型 | 模式 | 短提示logits逐位一致 | B8输入512/输出128 greedy | 分区计划 |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        plan=r['batch8_plan']
        lines.append(f"| {'0.8B' if r['model']=='small' else '9B'} | {r['mode']} | {r['bitwise_equal_steps']}/256 | {r['batch8_generation_tokens']}/1024 token一致 | {plan['splits']} × {plan['tokens_per_split']} |")
    lines+=['',f"全量测试{data['tests_passed']} passed。4组正式CLI逐步验证共{data['logits_steps_bitwise_equal']}个batch steps逐位一致；4组B8生成共{data['batch8_native_matching_tokens']}个token与固定原生基线一致。",'',
        '单元测试包含禁止原生decode回退、跨页和空行Graph replay、workspace地址稳定性、无效CLI配置及原始nano-vLLM路径拒绝matched选项。本轮kernel算法未改变。','',
        '短提示集8条，每条32步，以原生保存的完整logits作参考。B8使用固定seed927合成输入，对照前一轮同模型revision、同容量的原生token IDs；B8此次只检查完整生成token序列，不将其写成128步完整logits对照。','',
        '本轮将benchmark任务用于固定输入生成回归，repeats=1；正式性能结论仍由独立性能实验给出。原始runtime.json包含实际后端、分区与workspace。','',
        '用户命令和支持边界见根目录[RUN_QWEN35.md](../../RUN_QWEN35.md)。运行脚本及所有标量结果保留于results/entry-*；大模型权重不随交付包分发。','']
    (out/'RESULTS.md').write_bytes(('\n'.join(lines)).encode())
    print(json.dumps(data))


if __name__=='__main__':main()
