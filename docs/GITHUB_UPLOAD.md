# 用户手动上传 GitHub

本次未创建仓库，未自动commit或push。先由你在GitHub创建仓库；新建时建议保持空仓库。已有仓库也可使用下面工具，它不会重新初始化、改写提交历史、reset或force push。

**执行位置：本地 Windows PowerShell。** 先将 `$source` 指向本候选根目录；其余未知信息交互输入，不猜测姓名、邮箱和仓库地址。

```powershell
$source = (Read-Host '候选项目根目录的绝对路径')
$repo = Read-Host 'GitHub仓库URL（不含密码或token）'
$name = Read-Host '提交者姓名'
$email = Read-Host '提交者邮箱（可填GitHub提供的noreply邮箱）'
$target = Join-Path $HOME 'Documents\Codex\paged-decode-lab-github'
& (Join-Path $source 'release\Upload-GitHub.ps1') `
  -SourceDirectory $source -TargetDirectory $target `
  -RepositoryUrl $repo -AuthorName $name -AuthorEmail $email
```

脚本在独立checkout增量复制清单文件，不删除旧仓库额外文件。若目标已有工作区修改则立即停止并保留，先由你处理这些修改；脚本不自动stash。现有origin不读取或修改，你需人工确认目标正确。

每个Git调用立即保存并检查退出码；原始Git输出保留在终端。只对清单路径暂存，展示暂存文件后输入`COMMIT`才提交；提交成功且HEAD及当前分支有效后，输入`PUSH`才推送。拒绝或失败时保留checkout和暂存状态。

现有仓库中与候选同名的文件会形成可审阅修改，历史保留；旧仓库额外文件不自动删除，提交前检查它们是否仍适合公开。身份配置只写入目标仓库本地，不读登录凭据，不猜测代理设置。认证由你使用既有Git登录方式完成，禁止把token放进URL。

本次只做上传脚本语法和本地Git逻辑检查，不执行真实GitHub网络上传。

如果r2在Windows PowerShell 5.1下中文PDF文件名乱码中断，请使用r3并加`-ResumeVerifiedCopy`；它核验已有文件后继续。先加`-ValidateOnly`可只读预检。已有提交、暂存、额外文件或用户编辑会停止，不得删除工作区来强行绕过。详见[修复记录](RELEASE_R3.md)。
