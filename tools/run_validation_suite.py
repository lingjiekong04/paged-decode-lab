"""Validate all six model backend/mode combinations sequentially."""
import argparse
import itertools
from pathlib import Path
import subprocess
import sys
from _common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    baseline = destination / "flash_attn-eager"
    runs = []
    for backend, mode in itertools.product(("flash_attn", "triton_base", "triton_splitk"), ("eager", "graph")):
        name = f"{backend}-{mode}"
        cmd = [sys.executable, str(ROOT / "tools/validate_model.py"), "--model", args.model,
               "--backend", backend, "--mode", mode, "--tokens", str(args.tokens),
               "--output", str(destination / name)]
        if name != "flash_attn-eager":
            cmd += ["--reference", str(baseline)]
        with (destination / f"{name}.log").open("x", encoding="utf-8") as stream:
            result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        runs.append({"name": name, "exit_code": result.returncode})
        print(name, result.returncode, flush=True)
        if result.returncode:
            write_json(destination / "suite.json", {"status": "failed", "runs": runs})
            raise SystemExit(result.returncode)
    write_json(destination / "suite.json", {"status": "passed", "runs": runs})


if __name__ == "__main__":
    main()
