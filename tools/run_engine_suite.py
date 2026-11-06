"""Run all six backend/mode combinations sequentially in isolated processes."""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from _common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--kv-blocks", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    runs = []
    # Alternate eager/graph for each backend; a repeated campaign can reverse order.
    for backend, mode in itertools.product(("flash_attn", "triton_base", "triton_splitk"), ("eager", "graph")):
        name = f"{backend}-{mode}"
        cmd = [sys.executable, str(ROOT / "tools/benchmark_engine.py"), "--model", args.model,
               "--backend", backend, "--mode", mode, "--output", str(destination / name)]
        for key in ("batch", "input_tokens", "output_tokens", "kv_blocks", "repeats"):
            cmd += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        with (destination / f"{name}.log").open("x", encoding="utf-8") as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        runs.append({"name": name, "exit_code": result.returncode, "command": cmd})
        print(name, result.returncode, flush=True)
        if result.returncode:
            write_json(destination / "suite.json", {"status": "failed", "runs": runs})
            raise SystemExit(result.returncode)
    write_json(destination / "suite.json", {"status": "completed", "runs": runs})


if __name__ == "__main__":
    main()
