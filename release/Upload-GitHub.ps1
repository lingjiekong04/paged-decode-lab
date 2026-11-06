param(
    [Parameter(Mandatory=$true)][string]$RepositoryUrl,
    [Parameter(Mandatory=$true)][string]$TargetDirectory,
    [Parameter(Mandatory=$true)][string]$AuthorName,
    [Parameter(Mandatory=$true)][string]$AuthorEmail,
    [string]$SourceDirectory = (Split-Path -Parent $PSScriptRoot),
    [switch]$ResumeVerifiedCopy,
    [switch]$ValidateOnly,
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
# User runs this script. It is never executed by the release preparation agent.
if ($RepositoryUrl -notmatch '^(https://github\.com/|git@github\.com:)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$') {
    throw 'Use a GitHub repository URL without credentials.'
}
$source = (Resolve-Path -LiteralPath $SourceDirectory).Path
$target = [IO.Path]::GetFullPath($TargetDirectory)
if ($source -eq $target -or $target.StartsWith($source + [IO.Path]::DirectorySeparatorChar)) {
    throw 'Target must be a separate checkout outside the candidate directory.'
}
function Invoke-GitChecked {
    param([string[]]$GitArguments)
    & git @GitArguments
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "git failed with exit code $code; original error is printed above." }
}
function Get-Sha256 {
    param([string]$LiteralPath)
    $stream = [IO.File]::OpenRead($LiteralPath)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
# Windows PowerShell 5.1 defaults to ANSI. Manifest JSON is explicitly UTF-8.
# Validate the ENTIRE source before clone, copying, staging, or identity changes.
$manifest = Get-Content -LiteralPath (Join-Path $source 'RELEASE_MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$paths = @($manifest.files | ForEach-Object { $_.path }) + @('RELEASE_MANIFEST.json')
$currentHashes = @{}
foreach ($entry in $manifest.files) {
    $rel = [string]$entry.path
    if ([IO.Path]::IsPathRooted($rel) -or '..' -in ($rel -split '[/\\]')) { throw 'Unsafe manifest path.' }
    $from = Join-Path $source $rel
    if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { throw "Missing release file: $rel" }
    $actual = Get-Sha256 -LiteralPath $from
    if ($actual -ne $entry.sha256) { throw "Release hash mismatch: $rel" }
    $currentHashes[$rel] = $actual
}
$currentHashes['RELEASE_MANIFEST.json'] = Get-Sha256 -LiteralPath (Join-Path $source 'RELEASE_MANIFEST.json')
if (-not (Test-Path -LiteralPath $target)) {
    if ($ValidateOnly) { throw 'ValidateOnly requires an existing checkout; no clone attempted.' }
    Invoke-GitChecked -GitArguments @('clone', '--', $RepositoryUrl, $target)
} elseif (-not (Test-Path -LiteralPath (Join-Path $target '.git'))) {
    throw 'Existing target is not a Git checkout. Nothing was changed.'
}
$rootLines = & git -C $target rev-parse --show-toplevel
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Git root check failed ($code); preserve original output." }
if ([IO.Path]::GetFullPath(($rootLines -join '').Trim()) -ne $target) {
    throw 'Target is not the exact repository root.'
}
$status = & git -C $target status --porcelain=v1 --untracked-files=normal
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Git status failed ($code); preserve original output." }
if ($status) {
    if (-not $ResumeVerifiedCopy) { $status | Write-Host; throw 'Target has existing changes. Preserve them; use ResumeVerifiedCopy only for an interrupted release copy.' }
    # Resume is deliberately limited to an unborn, unstaged checkout.
    $head = & git -C $target rev-parse --verify --quiet HEAD
    $headCode = $LASTEXITCODE
    if ($headCode -eq 0) { throw 'Resume requires an empty repository with no commits. No files changed.' }
    if ($headCode -ne 1) { throw "HEAD check failed ($headCode); no files changed." }
    $tracked = & git -C $target status --porcelain=v1 --untracked-files=no
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "Tracked status failed ($code)." }
    if ($tracked) { $tracked | Write-Host; throw 'Staged/tracked changes found. No files changed.' }
    $previous = Get-Content -LiteralPath (Join-Path $source 'release\PREVIOUS_R2_FILES.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $previousHashes = @{}
    foreach ($entry in $previous.files) { $previousHashes[[string]$entry.path] = [string]$entry.sha256 }
    $pending = New-Object 'System.Collections.Generic.Stack[string]'
    $pending.Push($target)
    $verified = 0
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
            if ($directory -eq $target -and $item.Name -eq '.git') { continue }
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point in partial checkout; no files changed.' }
            if ($item.PSIsContainer) { $pending.Push($item.FullName); continue }
            $rel = $item.FullName.Substring($target.Length + 1).Replace('\', '/')
            if (-not $currentHashes.ContainsKey($rel)) { throw "Unrecognized existing file; no files changed: $rel" }
            $actual = Get-Sha256 -LiteralPath $item.FullName
            if ($actual -ne $currentHashes[$rel] -and $actual -ne $previousHashes[$rel]) { throw "Existing file has user edits; no files changed: $rel" }
            $verified++
        }
    }
    Write-Host "Verified $verified existing files against current/previous release. Resume is safe."
}
if ($ValidateOnly) { Write-Host 'Source and checkout validation passed; no files/config/staging/commit/push changed.'; return }
$remotes = & git -C $target remote
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Remote-name check failed ($code)." }
if (('origin' -notin @($remotes)) -and -not $PrepareOnly) {
    Invoke-GitChecked -GitArguments @('-C', $target, 'remote', 'add', 'origin', $RepositoryUrl)
}
# Do not read or rewrite an existing remote URL, which may contain credentials.
Write-Host 'Existing origin is preserved. Confirm it points to your intended repository before approving the push.'
foreach ($rel in $paths) {
    if ([IO.Path]::IsPathRooted($rel) -or '..' -in ($rel -split '[/\\]')) { throw 'Unsafe manifest path.' }
    $from = Join-Path $source $rel
    $to = [IO.Path]::GetFullPath((Join-Path $target $rel))
    if (-not $to.StartsWith($target + [IO.Path]::DirectorySeparatorChar)) { throw 'Path escaped checkout.' }
    if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { throw "Missing release file: $rel" }
    if ($rel -ne 'RELEASE_MANIFEST.json') {
        $expected = ($manifest.files | Where-Object { $_.path -eq $rel }).sha256
        if ((Get-Sha256 -LiteralPath $from) -ne $expected) {
            throw "Release hash mismatch: $rel"
        }
    }
    # Refuse reparse points in an existing checkout; never copy through symlinks.
    $inspect = Split-Path -Parent $to
    while ($inspect.Length -ge $target.Length) {
        if ((Test-Path -LiteralPath $inspect) -and ((Get-Item -LiteralPath $inspect).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Reparse point in target.' }
        if ($inspect -eq $target) { break }
        $inspect = Split-Path -Parent $inspect
    }
    if ((Test-Path -LiteralPath $to) -and ((Get-Item -LiteralPath $to).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Reparse point target file.' }
    New-Item -ItemType Directory -Path (Split-Path -Parent $to) -Force | Out-Null
    Copy-Item -LiteralPath $from -Destination $to -Force
}
if ($PrepareOnly) { Write-Host 'Release files copied; no identity/config/staging/commit/push changed.'; return }
Invoke-GitChecked -GitArguments @('-C', $target, 'config', '--local', 'user.name', $AuthorName)
Invoke-GitChecked -GitArguments @('-C', $target, 'config', '--local', 'user.email', $AuthorEmail)
# Only candidate paths are staged; existing unrelated tracked files are retained.
foreach ($rel in $paths) {
    Invoke-GitChecked -GitArguments @('--literal-pathspecs', '-C', $target, 'add', '--', $rel)
}
Invoke-GitChecked -GitArguments @('-C', $target, 'diff', '--cached', '--stat')
Invoke-GitChecked -GitArguments @('-C', $target, 'diff', '--cached', '--name-only')
Write-Host 'Inspect staged changes with: git -C <target> diff --cached'
$answer = Read-Host 'Type COMMIT to approve this staged candidate (anything else stops without deleting changes)'
if ($answer -cne 'COMMIT') { Write-Host 'Stopped; staged changes preserved.'; return }
Invoke-GitChecked -GitArguments @('-C', $target, 'commit', '-m', 'Prepare Paged Decode Lab open-source delivery')
Invoke-GitChecked -GitArguments @('-C', $target, 'rev-parse', '--verify', 'HEAD')
$branchLines = & git -C $target symbolic-ref --quiet --short HEAD
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Branch check failed ($code); no push attempted." }
$branch = ($branchLines -join '').Trim()
if (-not $branch) { throw 'No current branch; no push attempted.' }
$answer = Read-Host 'Type PUSH after checking origin and commit (anything else stops)'
if ($answer -cne 'PUSH') { Write-Host 'Commit retained; no push attempted.'; return }
Invoke-GitChecked -GitArguments @('-C', $target, 'push', '--set-upstream', 'origin', $branch)
