"""Run CPU checks without installing/importing torch, Triton, or model weights."""
import subprocess
import sys
import os
from _common import ROOT

if __name__ == "__main__":
    temp = ROOT / ".cache/test-tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTEST_DEBUG_TEMPROOT": str(temp)}
    # Select CPU files before collection so installed GPU packages are not imported.
    tests = ["tests/test_reference.py", "tests/test_vendor.py", "tests/test_downloader.py", "tests/test_splitkv_plan.py"]
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, env=env))
