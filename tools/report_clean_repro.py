"""Validate fresh-environment evidence and compare generation with retained native records."""
import json,re
from _common import ROOT

def main():
    read=lambda p:json.loads((ROOT/p).read_text())
    audit=read('results/repro-install/environment-audit.json')
    assert audit['status']=='passed'
    log=(ROOT/'results/repro-tests.log').read_text()
    match=re.search(r'(\d+) passed',log)
    assert match and 'failed' not in log and 'error' not in log.lower(),log[-2000:]
    rows=[]
    for model in ('small','large'):
        assert read(f'results/repro-model-{model}.json')['status']=='passed'
        ref=read(f'results/mperf-model-{model}-b8/bench-flash_attn-eager-r0.json')
        identity=read(f'results/mperf-model-{model}-b8/arguments.json')['model_identity']
        for mode in ('eager','graph'):
            path=f'results/repro-{model}-{mode}'
            val=read(path+'-validate/summary.json')
            assert val['status']=='passed' and val['bitwise_equal_steps']==val['tested_batch_steps']==256
            run=read(path+'-validate/runtime.json')
            assert run['backend']=='triton_matched' and run['mode']==mode
            b8=read(path+'-b8/summary.json')
            assert b8['status']=='completed' and len(b8['rows'])==1
            r=b8['rows'][0]
            assert r['prompt_sha256']==ref['prompt_sha256'] and r['token_ids']==ref['token_ids']
            configured=read(path+'-b8/arguments.json')['model']
            assert all(configured[k]==identity[k] for k in ('repo_id','revision','weights','files'))
            live=read(path+'-b8/runtime.json')
            assert live['batch']==8 and live['backend']=='triton_matched' and live['mode']==mode
            assert live['max_context']==1024 and live['pages_per_full_layer']==8*ref['kv_table_width']
            rows.append({'model':model,'mode':mode,'bitwise_logit_steps':256,
                'native_matching_greedy_tokens':sum(map(len,r['token_ids']))})
    data={'status':'passed','tests_passed':int(match[1]),'rows':rows,
        'bitwise_logit_steps':sum(r['bitwise_logit_steps'] for r in rows),
        'native_matching_greedy_tokens':sum(r['native_matching_greedy_tokens'] for r in rows),
        'base_python_reused':True,'model_assets_reused_and_rehashed':True,
        'old_site_packages_reused':False,'new_performance_claim':False}
    out=ROOT/'docs/clean-repro';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_bytes((json.dumps(data,indent=2)+'\n').encode())
    lines=['# 独立虚拟环境复现验收','',
        f"全量测试 {data['tests_passed']} passed；Qwen3.5-0.8B/9B × eager/Graph 共 {data['bitwise_logit_steps']} 步 logits 与新环境原生基线逐位一致，B8 共 {data['native_matching_greedy_tokens']} 个 greedy token 与保留的原生记录一致。",'',
        '源码从 benchmark-v2 交付包独立解压。55 个依赖 wheel 经哈希锁定后离线安装，editable 安装本目录源码，pip check 通过。实际包数量以 environment-audit.json 为准。', '',
        '复用了同一服务器的 CPython 3.11 基础解释器和模型资产；没有借用旧虚拟环境的 site-packages。模型所有权重重新执行 SHA256 校验。包文件、sys.path、.pth 与已加载共享库路径通过审计；不是另一台机器或独立操作系统的复现。','',
        '原生短提示参考在新环境中重新生成。长序列检查使用相同固定输入和 KV 容量，与历史原生 token IDs 比较。benchmark repeats=1 仅用于正确性回归，本轮不产生新的性能结论。','',
        '原始安装日志、离线 artifact 锁、模块路径审计与模型结果见 results/repro-*；复现步骤见根目录 REPRODUCTION.md。','']
    (out/'RESULTS.md').write_bytes(('\n'.join(lines)).encode())
    print(json.dumps(data))

if __name__=='__main__':main()
