"""Recompute released statistics from allow-listed historic measurements."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("resident", ROOT / "tools/report_fragment_resident.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
def load(name):
    return json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))

if __name__ == "__main__":
    expected = load("summary.json")
    assert report.operator_summary(load("operator.json")) == expected["operators"]
    for size in ("small", "large"):
        actual = report.model_summary(load(f"model-{size}.json"))
        assert actual == next(x["results"] for x in expected["models"] if x["size"] == size)
    print("Historical evidence recomputation passed: 480 operator + 160 model timing rows.")
    print("This check does not execute GPU kernels or regenerate measurements.")
