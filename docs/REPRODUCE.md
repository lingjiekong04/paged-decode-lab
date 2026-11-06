# 分层复现说明

所有命令均在**候选项目根目录**运行。Windows只提供CPU检查；真实GPU模型命令使用Linux。所有实验输出目录必须新建且不存在，不覆盖旧结果。

## 1. 无GPU最小示例与源码检查

Windows PowerShell：

```powershell
$env:PAGED_CPU_TEMP_ROOT = Join-Path $HOME 'Documents\Codex\.cache\paged-cpu'
python -m venv .venv-cpu
if ($LASTEXITCODE -ne 0) { throw 'venv failed' }
& .\.venv-cpu\Scripts\python.exe -m pip install -e '.[test]'
if ($LASTEXITCODE -ne 0) { throw 'pip failed' }
& .\.venv-cpu\Scripts\python.exe release/rebuild_upstream.py
if ($LASTEXITCODE -ne 0) { throw 'rebuild failed' }
& .\.venv-cpu\Scripts\python.exe release/check_cpu.py
if ($LASTEXITCODE -ne 0) { throw 'CPU tests failed' }
& .\.venv-cpu\Scripts\python.exe examples/cpu_attention.py
if ($LASTEXITCODE -ne 0) { throw 'demo failed' }
& .\.venv-cpu\Scripts\python.exe release/verify_public_results.py
if ($LASTEXITCODE -ne 0) { throw 'evidence failed' }
```

Linux终端：

```bash
set -e
python3 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -e '.[test]'
.venv-cpu/bin/python release/rebuild_upstream.py
.venv-cpu/bin/python release/check_cpu.py
.venv-cpu/bin/python examples/cpu_attention.py
.venv-cpu/bin/python release/verify_public_results.py
```

重建的固定基线ZIP SHA256为`8122b4981d58aa72be3fe30181e8cccf1f3aeeb6a5f2df833520d4a0c22a1a96`。联网由使用者主动执行；归档URL见upstream.json。已有同一归档时，加`--archive /path/to/baseline.zip`，无需联网。重建目标已存在会失败，避免重写；`test_vendor.py`随后核验全部交付文件。

`release/check_cpu.py`显式选择CPU模块，不导入GPU测试。原来的`tools/check.py`仅选择较早CPU子集，最新交付以release入口为准。CI执行同一入口；本次只验证本地执行，没有声称GitHub Actions已远端运行。

## 2. Linux GPU环境

推荐Python3.11、A100(sm80)、80GB。按`requirements-qwen35.txt`准备独立环境：Torch2.6.0、Triton3.2.0、FA2.7.4.post1、Transformers5.3.0。版本固定不消除CUDA／驱动／编译器ABI要求。FlashAttention可能需预先准备匹配wheel，或安装编译工具链；本次未执行新环境联网安装，也不提供未核验wheel下载链接。

```bash
set -e
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
# 使用者自行确认GPU环境与ABI后安装；可能触发扩展编译。
python -m pip install -r requirements-qwen35.txt
python -m pip install -e '.[test]'
```

## 3. 固定模型

| 模型 | revision |
|---|---|
| Qwen/Qwen3.5-0.8B | 2fc06364715b967f1860aea9cf38778875588b17 |
| Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a |

`configs/models.json`含各权重分片大小及SHA256。它们是远程项目现存记录，本次没有再次读取权重。本仓库不附模型；下载前检查上游模型许可和可访问性。已知元数据文件的完整哈希未全量收录于公开配置，下载工具会为本次实际文件生成manifest，不补造缺失哈希。

```bash
python tools/download_model.py --size small --output models/Qwen3.5-0.8B
python tools/download_model.py --size large --output models/Qwen3.5-9B
source tools/env.sh
```

先下载再source env.sh；该脚本设置离线运行变量。Windows可下载资源但不支持本项目CUDA执行路径。

## 4. 正式运行入口

```bash
python tools/qwen35.py --model models/Qwen3.5-0.8B --backend triton_matched \
  --mode eager --output results/demo-eager-new
python tools/qwen35.py --model models/Qwen3.5-0.8B --backend triton_matched \
  --mode graph --output results/demo-graph-new
```

默认flash_attn；显式支持base、splitk、tc、tc_splitk和固定容量matched。三组模型与提示必须使用同revision、容量和输出预算。strict validate先记录native eager基线，再用`--reference`比对；配置与参数见`python tools/qwen35.py --help`。新版S2不在该CLI后端列表中。

## 5. 最新实验复跑（需要GPU／模型，整理阶段未执行）

```bash
CUDA_VISIBLE_DEVICES=0 python tools/resident_experiment.py \
  --task correctness --output results/resident-correct-new
CUDA_VISIBLE_DEVICES=0 python tools/resident_experiment.py \
  --task model --model models/Qwen3.5-0.8B --output results/resident-small-new
CUDA_VISIBLE_DEVICES=0 python tools/resident_experiment.py \
  --task model --model models/Qwen3.5-9B --output results/resident-large-new
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_fragment_resident_models.py \
  --model models/Qwen3.5-0.8B --output results/resident-perf-small-new
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_fragment_resident_models.py \
  --model models/Qwen3.5-9B --output results/resident-perf-large-new
```

这些入口保持原始源码和预检，要求sm80、D256等条件，不做静默fallback。先确认选中GPU无其他计算进程；工具检测干扰时失败，不能自动杀进程。GPU工具会创建新输出和编译缓存，用户主动运行才会产生这些写入。

原`resident_experiment --task benchmark`硬编码读取`results/residency-snapshots-v2/n512-layer3.pt`和`n2048-layer3.pt`。这些是未公开的真实激活，不随仓库分发，没有公开下载地址。旧snapshot采集入口还依赖`results/stage-large-v1/inputs.json`。本候选不伪造快照或宣称这条命令可直接复跑；可离线重算480条已公开测量，或自行在相同模型上采集新数据并标为新实验，不能冒充历史逐字节复现。

## 6. 离线结果重算与PDF

```bash
python release/verify_public_results.py
python release/verify_manifest.py
```

公开数据是字段白名单派生结果。统计函数与远程源码字节一致，重算检查480条算子、160条模型计时及数值门槛。此命令不运行GPU。私有监控与receipt只在审查包保留，外部读者不能通过公开摘要独立验证全部原始运行过程。

PDF已交付，可直接阅读。重新生成需要reportlab和可合法使用的中文TrueType字体；不分发系统商业字体。`release/build_pdf.py --help`给出参数；渲染校验使用Poppler。重新生成PDF会改变发布清单，应产生新候选版本，不能重写历史清单来掩盖源码变化。

Windows沙箱可能限制OS临时目录的子进程访问，深层工程还可能触及MAX_PATH。上方PAGED_CPU_TEMP_ROOT把测试产物放在较短的独立可写路径；每次建立新目录，保留失败，不自动清理。普通Linux默认使用OS临时目录。
