# r3：Windows PowerShell 5.1 上传修复

这是发布工具修订，不是算法升级；继承的核心源码、配置、实验数据及PDF与r2逐字节相同。r2原清单和ZIP保留，不修改历史哈希。

## 实际问题

用户使用Windows PowerShell 5.1运行上传脚本。旧脚本Get-Content未指定编码，按本机默认ANSI解释UTF-8清单，导致中文PDF文件名乱码，随后报Missing release file。Git空仓库已clone成功，复制了14个文件，尚未暂存或提交。读取文件失败不是GitHub网络问题。

## 修复

- 读取JSON时显式使用-Encoding UTF8。
- 在clone或复制前，先验证全部源文件与SHA256，减少半完成输出。
- 新增-ResumeVerifiedCopy，仅用于没有HEAD、没有暂存／tracked修改的空仓库。现有文件必须属于候选清单，且内容哈希匹配当前或记录的r2原文件；发现额外文件、修改或符号链接则停止。没有删除或强制清理。
- 新增-ValidateOnly，只检查源和现有checkout；新增-PrepareOnly，只复制文件，均不提交或推送。
- 仍由用户输入COMMIT与PUSH决定实际提交和上传；已有仓库不改写历史或origin。

原始R2文件哈希保存在release/PREVIOUS_R2_FILES.json，仅用于识别未修改的中断复制产物，不认可任意用户修改。身份、仓库URL和认证不写进公开项目配置。

本轮使用实际Windows PowerShell 5.1检查中文清单、完整复制和中断恢复；用临时本地Git空仓库做验证，不做GitHub网络操作、commit或push。用户现有checkout仅做只读核验。具体结果见docs/checks/upload-r3.json。
