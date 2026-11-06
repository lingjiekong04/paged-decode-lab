"""Summarize D256 correctness and actual model regression; no latency claims."""
import json
from _common import ROOT


def main():
    out=ROOT/'docs/fragment-d256';out.mkdir(parents=True,exist_ok=True)
    c=json.loads((ROOT/'results/d256-correctness-v1/summary.json').read_text())
    assert c['status']=='passed'
    valid=[r for r in c['checks'] if 'fp32_close' in r]
    pairs=[r for r in c['checks'] if r.get('comparison')=='baseline_vs_shared']
    native=[r for r in valid if r['native_equal'] is not None]
    assert len(valid)==48 and len(pairs)==24 and len(native)==36
    assert all(r['native_equal'] for r in native)
    assert all(r[k]['bitwise_equal'] for r in pairs for k in ('output','partials','lse'))
    rows=[]
    for model,phase in [('0.8B','d256-small-v2'),('9B','d256-large-v1')]:
        data=json.loads((ROOT/'results'/phase/'summary.json').read_text());assert data['status']=='passed'
        selected=[r for r in data['checks'] if r['backend']=='shared']
        steps=[s for r in selected for s in r['steps']]
        free=[r for r in data['free_checks'] if r['backend']=='shared']
        assert len(steps)==640 and len(free)==20 and all(s['bytes_equal'] and s['argmax_equal'] for s in steps)
        assert all(r['tokens_equal'] for r in free)
        rows.append({'model':model,'revision':data['model']['revision'],'shared_logit_steps':len(steps),'shared_greedy_runs':len(free),
                     'all_recorded_steps':data['logit_steps'],'all_free_runs':data['free_runs'],'phase':phase,
                     'full_attention_layers':data['full_attention_layers']})
    summary={'models':rows,'operator_outputs':len(valid),'operator_pairs':len(pairs),'native_checks':len(native),
             'native_equal':sum(r['native_equal'] for r in native),'cpu_tests':7,'performance_claim':False,'production_defaults_changed':False}
    (out/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
    text='''# D256扩展与Qwen3.5真实模型回归

实验接口新增D256/TILE64，保留D64/128的生成字符串和正式默认路径。新的host适配器为 [fragment_d256_common.py](../../tools/fragment_d256_common.py)，仍使用此前保序partition的通用计算体；老接口继续按原范围拒绝D256，避免改变历史交付与实验协议。

## 算子验收

sm80、4 warps、MMA-v2 `[1,4]`，TILE64下每线程8个FP32概率槽位，共1024个逻辑位置；GPU标记探针无遗漏、重复或映射错误，空上下文CTA未写probe。共享暂存4096 bytes、四状态256 bytes，合计静态shared4352 bytes。预检动态shared79232 bytes；原→候选的spill字段22→8、STACK88→32 bytes。这是该预检配置的静态资源记录，不是所有形状或模型占用率结论。

48次FP32/Graph输出检查、24组output/partials/LSE严格字节对照通过，36次非空原生FlashAttention对照通过。GQA group1/4/16，默认split与单split，零上下文、1/255/256/257/1023/1024/1025，增长到2051，随机物理页、-1补齐及NaN尾部、页表翻转与KV更新均覆盖。保留首项alpha收缩为FMA，Graph捕获之前完成编译/布局/共享预算门槛；没有fallback。

新增7项CPU测试验证D256位置覆盖、D64/128生成代码不变、D256收缩指令及错误tile拒绝。原始算子结果见 [correctness](../../results/d256-correctness-v1/summary.json) 与 [smoke](../../results/d256-smoke-v2/summary.json)。

## 模型实验入口

[regress_fragment_d256_model.py](../../tools/regress_fragment_d256_model.py) 在独立进程注册实验AttentionInterface回调，不修改HybridRunner及正式CLI默认后端。prefill仍走原生FlashAttention，线性注意力仍走固定Transformers官方Torch fallback；仅decode的full-attention层可切换为新partition。

模型加载后，每个full-attention层预分配资源并prepare；固定query缓冲copy、KV写入、线性状态更新和lm_head均位于完整decode Graph执行中。eager每步使用同一实验实现。旧matched路径和原生路径作为控制，三个路径各测eager/Graph。

每个模型为8个固定提示＋255/1023 token边界输入，共10个提示；每提示固定输出32 token，忽略EOS提前停止。先在相同原生token前缀上比较每一步完整logits的字节及argmax，然后独立free greedy重跑并对照token序列。每步记录logits SHA256、最大误差和RMSE；如有分歧会保存首个差异张量并立即停止。

| 模型 | 固定revision | 新路径logits步骤 | 新路径free greedy | 模式 |
|---|---|---:|---:|---|
'''
    for r in rows:text+=f"| Qwen3.5-{r['model']} | `{r['revision']}` | {r['shared_logit_steps']} | {r['shared_greedy_runs']} | eager / Graph |\n"
    text+='''
两模型的新路径合计1280个logits步骤逐字节一致、40次free greedy序列一致；原生与原matched控制也保留完整逐步记录。每模型共记录1920步，其中320步是native-eager参考自身，不计作新路径验收。first token由prefill产生，因此各新路径640步中20步属于prefill参考，其余620步为真实decode；两模型共1240步新decode logits。

模型身份与文件清单沿用固定manifest，实验记录revision、输入token IDs、源码SHA、设备及库版本。大模型为实际Qwen3.5-9B，并非8B重命名。完整结果：[0.8B](../../results/d256-small-v2/summary.json)、[9B](../../results/d256-large-v1/summary.json)。

## 边界与复现

本轮服务器有其他常驻CUDA上下文，未干预它们；共享状态保存在arguments.json。只报告数值和生成回归，不报告独占性能、模型吞吐或端到端加速。32 token、固定提示与两个跨页长度不是任意长上下文/批次的等价证明。模型batch为1，最大容量1280；后续性能验收需空闲GPU和独立重复实验。

首次smoke被GPU独占PID门槛拦下，随后明确改为共享环境下的正确性任务；首次模型启动遇到tokenizer返回BatchEncoding，统一提取input_ids后重跑。失败日志及当时源码保留，均未作为通过结果。

```bash
source tools/env.sh
export CUDA_VISIBLE_DEVICES=0
/path/to/venv/bin/python tools/smoke_fragment_d256.py --output results/new-d256-smoke
/path/to/venv/bin/python tools/ablate_fragment_d256.py --task correctness --output results/new-d256-correctness
/path/to/venv/bin/python tools/regress_fragment_d256_model.py --model models/Qwen3.5-0.8B --output results/new-d256-small
/path/to/venv/bin/python tools/regress_fragment_d256_model.py --model models/Qwen3.5-9B --output results/new-d256-large
/path/to/venv/bin/python -m pytest tests/test_fragment_d256.py -q
```

完成了D256到真实模型回归闭环；实验回调尚未升为默认后端。下一项工作是空闲GPU上的模型性能验收，再决定是否增加正式入口选项。
'''
    (out/'RESULTS.md').write_bytes(text.encode())
    (ROOT/'FRAGMENT_D256.md').write_bytes('''# D256与真实模型回归

[报告](docs/fragment-d256/RESULTS.md) · [汇总](docs/fragment-d256/summary.json)

D256/TILE64分页、GQA、Graph及字节对齐通过。Qwen3.5-0.8B/9B完成实验后端eager/Graph逐步logits及独立greedy回归，正式默认入口保持原样。本轮共享GPU，不宣称模型加速。
'''.encode())
    print(json.dumps(summary))


if __name__=='__main__':main()
