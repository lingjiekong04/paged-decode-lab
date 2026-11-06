# 系统架构与实现边界

## 两条集成路径

第一条是固定 nano-vLLM：上游提供 Qwen3 模型、权重加载、调度、TP、prefill、KV 写入和 Graph 框架；本项目在 decode attention 分支接入自定义后端，并增加 workspace 与配置校验。上游基线由 `upstream.json` 固定，发布包只包含覆盖层和原始补丁。

第二条是 Qwen3.5 hybrid runner：`HybridRunner` 调用 Transformers 5.3.0 的官方模型块，把 full-attention decode 接入分页 KV 和自定义后端。linear-attention 层继续调用官方 Gated DeltaNet，维护 conv/recurrent state。它不等同于 nano-vLLM 原生支持 Qwen3.5，不提供连续批处理或多卡通信。

## 张量与寻址

- Q：`[batch, query_heads, head_dim]`。
- K/V：NHD 物理页，`[physical_pages, 256, kv_heads, head_dim]`。
- table：`int32[batch, table_capacity]`；lengths：`int32[batch]`。
- 输出：借用的 BF16 `[batch, query_heads, head_dim]`；下次调用覆盖，不可并发跨 stream 共用同一 workspace。

对逻辑位置 t，页号 `p=table[b,t//256]`，页内槽 `s=t%256`。地址偏移为 `((p*256+s)*Hkv+kv_head)*D+d`。长度 mask 必须在无效表项或 KV 尾部读取前生效。debug 元数据内容校验会同步 CPU，只能在热路径及 Graph capture 之外调用；热路径校验 shape/dtype 不代表外部页表内容安全。

## 数值方案

基础版维护 online softmax 的最大值 m、分母 l 和加权和 a。新 tile 的最大值 m' 确定后，用 `exp(m-m')` 缩放历史状态，再累加当前概率。FP32 累加减小误差，最终转 BF16。

固定1024-token split-K把序列分为多个分区，输出每区 m/l/a，merge用全局最大值重新缩放。空分区贡献零，全空输出零，避免 `-inf - -inf` 和除零。

`matched_splitkv.py` 是不同的数值协议：按固定容量规划 splits，保存 normalized FP32 partial 与 LSE，再稳定合并。BF16 QK/PV、分母逐 lane 的四状态累加、log 指令与 FMA 收缩都会影响舍入。容差内的 attention 误差仍可能改变接近的 logits argmax，因此保留算子参考、完整 logits 和自由生成三层验证。

## Graph 生命周期

1. 固定 batch、最大上下文、页表完整宽度和后端。
2. 分配输出、workspace、KV 及线性层状态，保持地址稳定。
3. 编译并预热；实验后端检查 MMA 布局、寄存器/shared和驱动驻留上限。
4. 捕获文本 decode forward，包含 KV 写入、linear state 更新、lm_head。
5. 在稳定地址中更新 query/KV/table/lengths等内容，再 replay。

prefill和token选择在Graph外。捕获成功不是replay正确性的证明，还需验证零行、页增长和多轮内容更新。生产默认 `flash_attn` 未被最新实验替换。

## D256 两阶段实验

入口 `tools/fragment_resident_common.py` 复用冻结 partition，唯一手动流水线配置变化为 stages 3→2。shared方案使用4096 B概率暂存和256 B分母状态，共4352 B静态shared。预检限定sm80、D256、TILE64、4warps及实际MMA布局，检查共享声明及驱动至少2CTA/SM；不满足则报错，无静默回退。

驱动查询的是理论可驻留上限，不能写成实测 achieved occupancy。寄存器／spill也随调度改变，无法把全部收益只归因于shared容量；没有新硬件计数器证据。
