"""Six configurations; retain numerical mismatches and stop on runtime errors."""
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
    parser.add_argument("--task", choices=("validate", "benchmark"), required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-tensorcore", action="store_true", help="add triton_tc eager/Graph (eight configurations)")
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    runs = []
    backends = ("flash_attn", "triton_base", "triton_splitk") + (("triton_tc",) if args.include_tensorcore else ())
    for backend, mode in itertools.product(backends, ("eager", "graph")):
        name = f"{backend}-{mode}"
        cmd = [sys.executable, str(ROOT / "tools/qwen35.py"), "--model", args.model,
               "--task", args.task, "--backend", backend, "--mode", mode,
               "--output", str(destination / name)]
        for key in ("tokens", "batch", "input_tokens", "repeats"):
            cmd += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        if args.task == "validate" and name != "flash_attn-eager":
            cmd += ["--reference", str(destination / "flash_attn-eager")]
        with (destination / f"{name}.log").open("x", encoding="utf-8") as stream:
            result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        runs.append({"name": name, "exit_code": result.returncode, "command": cmd})
        print(name, result.returncode, flush=True)
        if result.returncode not in (0, 2) or args.task == "benchmark" and result.returncode != 0:
            write_json(destination / "suite.json", {"status": "runtime_failure", "runs": runs})
            raise SystemExit(1)
    status = "completed" if all(r["exit_code"] == 0 for r in runs) else "numerical_mismatch"
    write_json(destination / "suite.json", {"status": status, "runs": runs,
               "note": "numerical_mismatch is a strict regression failure; no thresholds were relaxed"})
    raise SystemExit(0 if status == "completed" else 2)


if __name__ == "__main__":
    main()
