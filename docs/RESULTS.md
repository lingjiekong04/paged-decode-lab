# 实验结果、负结果与证据层级

## 当前主结果

详细历史报告原文复用并调整图片路径：[D256流水线报告](RESIDENT_PIPELINE_RESULTS.md)。该报告中的原始文件位置属于私有实验环境；公开版只提供经过白名单提取的 `evidence/` 数据，不将其冒充原始日志。

主实验为A100-SXM4-80GB，sm80，108SM；使用固定Torch/Triton/FA/Transformers版本。比较original（匹配数值协议的Triton基线）、shared_s3（此前共享归约）和shared_s2（两阶段）；nativeFA在独立模型实验中列出。

| 配置 | dynamic shared B | static shared B | 寄存器/线程 | Triton spill字段 | 驱动CTA/SM上限 |
|---|---:|---:|---:|---:|---:|
| original | 79232 | 0 | 255 | 16 | 2 |
| shared_s3 | 79232 | 4352 | 255 | 8 | 1 |
| shared_s2 | 77952 | 4352 | 192 | 0 | 2 |

驱动还为每CTA保留1024B，S2总计83328B/CTA，两个166656B，小于每SM167936B。这里的spill字段不是实测动态spill流量。

长上下文2064，partition同轮S3/S2中位数1.3686，完整算子1.2483；完整算子比original为1.1819。短上下文528，完整算子S3/S2为0.9762，对应S2时延增加约2.44%。五轮范围是观测min/max，不是置信区间。

480条算子记录使用此前保存的9B第3层、第16个decode位置的真实Q/K/V，输入512/2048、上下文528/2064、容量2112、36splits。每个Graph unroll32或128，replay20次，五轮随机交错。缩减grid只计算部分partial，不是完整attention；主表取完整36分区。固定输入反复访问属于hot-cache诊断，计时含节点间隙，不等于纯kernel busy time。JIT、预热、capture和workspace分配不在区间内。

## 模型与数值

Qwen3.5-0.8B／9B固定revision；batch1，输入512/2048，输出64，共160条wall记录。wall生成和带逐步event的生成分开测量；事件记录63个decode区间。三组自定义workspace始终保留，包括native配置，因此显存结果不代表native最小用量。两张GPU各自空闲运行，主机CPU共享；监测未发现同GPU外部计算进程。

两模型Graph wall同轮S3/S2中位数0.9984–1.0015，不能宣称显著端到端提升。eager部分波动较大，不用单个中位数推广。linear attention使用官方Torch fallback，不代表工业推理引擎全部优化后的性能。

分页正确性72项，保留FP32参考阈值atol=rtol=0.02，并验证Graph/eager及native。每模型10条自然／边界输入、3后端×2模式×32步=1920个完整logits步骤、60次自由生成；最新性能输入每模型另1024步完整logits对齐。结果仅对已测试输入和版本成立。

## 必须保留的负结果

| 阶段 | 失败／边界 | 后续处理 |
|---|---|---|
| 基础／固定split-K | 历史固定模型回归分别3/256、2/256个argmax分歧 | 保留失败；后续协议对齐是新路径 |
| TC ordered v2-gather | 使用固定Triton版本不存在的API，编译失败 | 不纳入有效性能候选 |
| 固定1024-token TC split-K | 9B长输入仍有token分歧 | 调查容量切分、partial/LSE与log协议 |
| 匹配协议v1 | 已记录激活的LSE仍不同，影响最终BF16输出 | v2控制快速log指令；未将诊断替换值用于正式输出 |
| 分母微内核接入 | 首次部分FP32 partial因FMA收缩不同而失败 | 明确接收未缩放状态和alpha并保留FMA |
| shared_s3 D256 | 静态shared使驻留上限降至1 CTA/SM | 显式S2减少dynamic shared，重跑同套回归 |
| shared_s2短上下文 | 完整算子约慢2.44% | 保留S3/S2显式选择，没有自动调优默认替换 |
| Nsight计数器 | 多种探测超时／无kernel采样 | 没有可用计数器，不能断言权限被拒绝 |

本表较早阶段根据远程现存报告和算法快照复核；本次未采集全部历史原始数据，也未重新运行。其报告原字节哈希见 `provenance/EXTRACTION_RULES.json`，不混入最新测试的通过数量。

最新同源控制路径重新编译后的.text与更早归档不完全相同。未确认原因；所有比值取同一campaign配对，禁止跨campaign比较绝对时延并声称仅一个编译参数决定全部变化。

## 本次可以复现到哪里

- 无GPU：CPU算法／协议测试、来源哈希、上游重建、历史公开统计重算。
- 有匹配GPU与依赖：随机分页kernel正确性、真实模型运行、最新自然提示回归及模型性能实验。
- 原480条真实张量算子计时：需要私有激活快照；本仓库不分发，不能直接运行原`--task benchmark`并声称完全复现。快照采集还依赖历史输入记录，缺口详见复现文档。
- 没有真实业务数据、服务SLA、在线负载或生产验收，不能推导业务效果或生产可用性。
