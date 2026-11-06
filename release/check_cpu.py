"""Explicit CPU collection; do not import torch/Triton or collect GPU tests."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_FILES = {
    "test_gpu.py", "test_hybrid.py", "test_precision.py",
    "test_matched_splitkv.py", "test_denominator_selection_gpu.py",
}

if __name__ == "__main__":
    if not (ROOT / "vendor/nano-vllm/LICENSE").is_file():
        raise SystemExit("First run: python release/rebuild_upstream.py [--archive baseline.zip]")
    tests = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "tests").glob("test_*.py"))
             if p.name not in GPU_FILES]
    # A short fresh OS-temp parent avoids Windows MAX_PATH when the project
    # is unpacked deeply. pytest owns only the initially absent child.
    # Keep each run directory for diagnosis; do not delete previous attempts.
    override = os.environ.get("PAGED_CPU_TEMP_ROOT")
    if override:
        Path(override).mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="pdc-", dir=override))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--basetemp", str(run / "t"), *tests], cwd=ROOT, env=env))
