# 求职展示与面试讲解

## 可使用的项目介绍

基于Triton实现分页Decode Attention和split-K，接入固定nano-vLLM及调用官方模型块的Qwen3.5执行路径，完成CUDA Graph和真实模型数值回归。分析分区、LSE、log与FMA舍入对生成一致性的影响，并通过显式流水线实验降低D256资源占用。

## 可使用的性能描述

在A100、固定9B真实attention激活、上下文2064的同轮五次对照中，将shared实验kernel的num_stages从3调整为2，完整partition+merge算子加速约1.25倍；短上下文约退化2.44%，模型端到端收益很小。该比较对象是本项目先前shared实现，不是FlashAttention。

## 建议演示顺序

1. 运行CPU示例解释逻辑位置到物理页映射，明确这不是GPU演示。
2. 打开kernels.py说明m/l/acc的online softmax和split合并。
3. 打开matched_splitkv.py说明为何容差内误差仍会改变greedy token。
4. 打开fragment_resident_common.py说明布局预检、静态／动态shared及S2/S3。
5. 运行公开统计重算，展示收益和退化并存，而非宣称全面超越工业算子。
6. 有GPU与合法模型资源时另演示真实CLI；否则出示带来源哈希的历史报告，不能假装正在推理。

## 应能回答的问题

- 为什么split边界与物理页边界不同？GQA头如何映射？
- 为什么小batch长KV可能需要split-K，短KV反而被merge或调度开销拖慢？
- 为什么数值更精确的log函数不一定与原生BF16输出一致？
- Graph需要稳定哪些地址，哪些数据可以replay前更新？
- 为什么驱动2CTA/SM上限不等于实测occupancy？
- 为什么算子1.25倍收益没有同等传递到模型吞吐？

禁止表述为从零实现整个Qwen推理引擎、自研Gated DeltaNet、实现多卡通信或全面超过FlashAttention。个人贡献包括AI辅助实现与分析，框架和模型块来自上游。
