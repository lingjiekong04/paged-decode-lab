"""Rebuild the two-model regression and three-shape operator report on CPU."""
import argparse
import json
import re
import statistics
from pathlib import Path
from _common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-e2e", action="store_true", help="also summarize fresh native/TC model runs")
    args = parser.parse_args()
    read = lambda name: json.loads((args.results / name).read_text(encoding="utf-8"))
    models, performance = {}, []
    for size in ("small", "large"):
        report = read(f"tensorcore-integrated-{size}/summary.json")
        if report["status"] != "passed" or not report["production_backend"]:
            raise ValueError("formal backend regression did not pass")
        cases = []
        for case in report["cases"]:
            reports = read(f"tensorcore-integrated-{size}/triton_tc-{case['mode']}/summary.json")["prompts"]
            steps = [s for r in reports for s in r["steps"]]
            if len(steps) != 256:
                raise ValueError("expected fixed eight-prompt, 32-step regression")
            cases.append({**case, "tested_steps": len(steps),
                "bitwise_equal_steps": sum(r["bitwise_equal"] for r in steps),
                "worst_logit_abs": max(r["max_abs"] for r in steps)})
        models[size] = cases
        rows = read(f"tensorcore-operator-{size}/summary.json")["rows"]
        if len(rows) != 75:
            raise ValueError("expected quick operator matrix with five backends and five repeats")
        for batch, length in ((1, 512), (1, 8192), (8, 2048)):
            times = {}
            for backend in ("flash_attn", "triton_base", "triton_splitk", "triton_tc", "tensorcore_bf16"):
                selected = [r for r in rows if (r["batch"], r["context"], r["backend"]) == (batch, length, backend)]
                if len(selected) != 5:
                    raise ValueError("missing repeats")
                profile = read(f"tensorcore-operator-{size}/profile-b{batch}-l{length}-d256-{backend}.json")
                times[backend] = {
                    "graph_event_us": statistics.median(r["graph_event_us"] for r in selected),
                    "graph_event_min_us": min(r["graph_event_us"] for r in selected),
                    "graph_event_max_us": max(r["graph_event_us"] for r in selected),
                    "full_call_wall_median_us": statistics.median(r["full_call_summary_us"]["median"] for r in selected),
                    "profiler_cuda_busy_us_per_call": sum(r["duration_us"] for r in profile["cuda_events"]) / profile["iterations"]}
            performance.append({"model_geometry": size, "batch": batch, "context": length, "latency": times,
                "tc_vs_base_graph_speedup": times["triton_base"]["graph_event_us"] / times["triton_tc"]["graph_event_us"],
                "tc_vs_flash_graph_ratio": times["triton_tc"]["graph_event_us"] / times["flash_attn"]["graph_event_us"],
                "ordered_vs_unordered_graph_ratio": times["triton_tc"]["graph_event_us"] / times["tensorcore_bf16"]["graph_event_us"]})
    match = re.search(r"(\d+) passed in", (args.results / "tensorcore-full-tests.log").read_text())
    if not match:
        raise ValueError("missing successful full test log")
    summary = {"status": "completed", "backend": "triton_tc", "gpu_tests": match[1] + " passed",
        "model_regression": models, "operator_performance": performance, "operator_repeat_records": 150,
        "methodology": "Graph event launch intervals and synchronized full-call timing; separate profiler run. Five repeats per shape/backend."}
    if args.include_e2e:
        rows = []
        for size in ("small", "large"):
            report = read(f"tc-e2e-{size}/summary.json")
            if report["status"] != "completed" or len(report["cases"]) != 4:
                raise ValueError("missing native/TC end-to-end cases")
            arguments = read(f"tc-e2e-{size}/arguments.json")
            budget = set()
            cases = []
            for case in report["cases"]:
                if len(case["rows"]) != 5:
                    raise ValueError("five end-to-end repeats required")
                if any(r["prompt_sha256"] != report["prompt_sha256"] for r in case["rows"]):
                    raise ValueError("benchmark inputs changed")
                budget.update(r["kv_pages_per_full_layer"] for r in case["rows"])
                cases.append({k: case[k] for k in ("backend", "mode", "throughput", "greedy_matches_native_all_repeats")})
            if len(budget) != 1:
                raise ValueError("KV capacity changed between benchmark cases")
            rows.append({"model_geometry": size, "batch": arguments["batch"], "input_tokens": arguments["input_tokens"],
                "output_tokens_per_sequence": arguments["tokens"], "kv_pages_per_full_layer": next(iter(budget)), "cases": cases})
        summary["end_to_end"] = rows
        summary["end_to_end_repeat_records"] = 40
        summary["end_to_end_note"] = "Fresh native/TC eager/Graph runs, same model instance/input IDs/KV capacity. Wall time includes prefill, decode, sampling and conversion. Token equality is separate from performance."
    write_json(args.output, summary)


if __name__ == "__main__":
    main()
