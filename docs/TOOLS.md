# 工具分级与依赖边界

| 类型 | 推荐入口 | 需要资源 |
|---|---|---|
| CPU最小示例 | examples/cpu_attention.py | NumPy，合成数据，无GPU |
| 上游重建／CPU测试 | release/rebuild_upstream.py、check_cpu.py | 固定ZIP或网络，pytest/NumPy |
| 公开统计重算 | release/verify_public_results.py | Python标准库，公开evidence |
| 正式模型CLI | tools/qwen35.py | Linux CUDA、固定模型与GPU依赖 |
| 最新分页回归 | tools/resident_experiment.py --task correctness | sm80、Torch/Triton/FA |
| 最新模型数值回归 | 同工具 --task model | 另需固定Qwen3.5权重 |
| 最新模型性能 | tools/benchmark_fragment_resident_models.py | 同上，空闲GPU |
| 原真实张量算子计时 | resident_experiment --task benchmark | 未公开历史KV快照，不能直接复跑 |
| 历史研究工具 | 其他ablate/probe/profile/report/export脚本 | 可能依赖对应原始results、IR或旧实验配置 |

为了保留最新工具的完整本地导入链，本候选继承tools源码。保留脚本不代表该脚本依赖的全部历史数据已公开；历史报告生成器通常读取原`results/`，应在私有原工程或配齐对应证据的新副本运行，不能把缺失数据替换成模拟结果。

当前推荐主线：CPU演示 → 基础kernel → split-K → matched协议 → 模型CLI → S2实验。目录不代表算法功能均已接入同一个生产后端，尤其shared_s2只在独立实验路径中启用。

`profile_partition_cubin_host.c`是加载Triton cubin的宿主探测器，不是自研CUDA C++ attention kernel。Nsight工具存在也不代表已得到有效动态计数器。
