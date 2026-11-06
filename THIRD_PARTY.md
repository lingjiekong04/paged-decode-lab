# 来源与贡献边界

- `vendor/nano-vllm/` 来自 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，MIT，固定提交 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。原始许可证保存在该目录。源文件哈希、修改文件与下载包 SHA256 见 `upstream.json`；相对上游的修改见 `patches/nano-vllm.patch`。
- 新增 Triton kernel、运行时、测试与实验工具由本项目编写，使用 AI 辅助。算法学习参考 [Triton fused attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)、[FlashAttention 论文](https://arxiv.org/abs/2205.14135)及 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)。没有复制或分发 FlashInfer 的源码。
- 上游承担模型实现、权重加载、RoPE、prefill、KV 写入、调度、TP 和 Graph 框架。本项目承担自定义 paged decode 计算、split-K 合并、后端接入、workspace、greedy 支持与实验工具；不能表述为从零编写整个推理引擎。
- PyTorch、Triton、FlashAttention、Transformers 是运行依赖，遵循各自许可，不随交付包分发二进制。
- 实验性精度路径参考 FlashAttention 2.7.4.post1 的 [softmax.h](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/softmax.h)、[utils.h](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/utils.h) 与 SM80 accumulator 布局理解 BF16 指数权重、分母的逐 lane 累加和归一化次序；新增 Triton 实现由本项目编写，没有分发其 C++ kernel。上游 FlashAttention 为 BSD-3-Clause，版权归原作者。
- Qwen 模型使用官方模型仓库，revision 见 `configs/models.json`，权重不随源码分发，使用时遵循其模型许可证。
- 实验性 `splitkv_plan.py` / `matched_splitkv.py` 根据固定 FlashAttention 2.7.4.post1 的 [flash_api.cpp](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/flash_api.cpp)、[flash_fwd_kernel.h](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/flash_fwd_kernel.h) 和上述 softmax 文件，理解容量分区、静态 occupancy 启发式、归一化 partial/LSE 协议及合并次序。实现以 Python/Triton 重写，并保留上游为 BSD-3-Clause、版权归原作者的来源说明；没有分发其 C++ attention kernel。
- Qwen3.5 runner 组合调用 Transformers 5.3.0 的官方模型块，参考其文本 forward 数据流；Gated DeltaNet、RoPE、norm、MLP、gate 与模型权重均归上游。对应上游文件为 `transformers/models/qwen3_5/modeling_qwen3_5.py`（Apache-2.0）。本项目没有把上游 linear attention 算法表述为自研。
- 选题流程参考 [shushu-internship-tool](https://github.com/LiuMengxuan04/shushu-internship-tool)（Apache-2.0）；没有复制其脚本，也没有伪造脚本评分。
