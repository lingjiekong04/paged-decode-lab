"""Rebuild TC split-K numerical and performance report from raw scalar evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
from _common import ROOT, write_json


def build_summary(base):
    read = lambda name: json.loads((base / name).read_text(encoding="utf-8"))
    diagnosis, models, operators = [], [], []
    for size in ("small", "large"):
        prefix = f"long-diagnosis-{size}"
        meta, tc, rescue, audits = (read(prefix + "/" + name) for name in
            ("summary.json", "tc.json", "tc_rescue_auto.json", "same-input.json"))
        selected = [r for r in tc["attention_inputs_outputs"] if r["location"].endswith("/attention3")]
        if not selected:
            raise ValueError("missing first-full-attention inputs")
        diagnosis.append({"model": size, "target_step": meta["target_step"], "sequence": meta["target_sequence"],
            "batch": meta["batch"], "traced_batch_steps": len(tc["steps"]),
            "tc_logits_equal_native_single": all(r["vs_native_single"]["bitwise_equal"] for r in tc["steps"]),
            "first_different_layer": next((r["location"] for r in tc["layers"] if not r["bitwise_equal"]), None),
            "first_full_qkv_equal": all(all(r[k]["bitwise_equal"] for k in ("q", "k", "v")) for r in selected),
            "rescue_logits_equal_auto": rescue["all_logits_equal_auto"],
            "same_input_samples": len(audits),
            "same_input_tc_equals_single": all(r["tc_vs_native_single"]["bitwise_equal"] for r in audits),
            "same_input_fp32_pass": all(r["tc_vs_fp32"]["close_2e-2"] for r in audits),
            "target_batch_logit_error": tc["steps"][-1]["vs_native_auto"]})
        run = read(f"tcsplit-model-{size}/summary.json")
        validation = read(f"tcsplit-model-{size}/validation.json")
        budgets, hashes = set(), set()
        for case in run["benchmark"]:
            for repeat in range(5):
                row = read(f"tcsplit-model-{size}/bench-{case['backend']}-{case['mode']}-r{repeat}.json")
                budgets.add(row["kv_pages_per_full_layer"])
                hashes.add(row["prompt_sha256"])
        if len(budgets) != 1 or len(hashes) != 1:
            raise ValueError("benchmark input/KV budget differs between cases")
        graph = {r["backend"]: r["throughput"]["median"] for r in run["benchmark"] if r["mode"] == "graph"}
        models.append({"model": size, "validation": validation, "benchmark": run["benchmark"],
            "max_context": run["max_context"], "partition_capacity": run["partition_capacity"],
            "kv_pages_per_full_layer": next(iter(budgets)),
            "split_vs_tc_graph_throughput_ratio": graph["triton_tc_splitk"] / graph["triton_tc"],
            "split_vs_native_graph_throughput_ratio": graph["triton_tc_splitk"] / graph["flash_attn"]})
        rows = read(f"tcsplit-operator-{size}/summary.json")["rows"]
        if len(rows) != 75:
            raise ValueError("expected three shapes, five backends, five repeats")
        for batch, length in ((1, 512), (1, 8192), (8, 2048)):
            latency = {}
            for backend in ("flash_attn", "triton_base", "triton_splitk", "triton_tc", "triton_tc_splitk"):
                group = [r for r in rows if (r["batch"], r["context"], r["backend"]) == (batch, length, backend)]
                if len(group) != 5:
                    raise ValueError("missing operator repetitions")
                profiler = read(f"tcsplit-operator-{size}/profile-b{batch}-l{length}-d256-{backend}.json")
                latency[backend] = {"graph_us": statistics.median(r["graph_event_us"] for r in group),
                    "graph_min_us": min(r["graph_event_us"] for r in group),
                    "graph_max_us": max(r["graph_event_us"] for r in group),
                    "full_call_wall_median_us": statistics.median(r["full_call_summary_us"]["median"] for r in group),
                    "profiler_busy_us": sum(r["duration_us"] for r in profiler["cuda_events"]) / profiler["iterations"],
                    "workspace_bytes": group[0]["workspace_bytes"],
                    "max_abs_vs_fp32": max(r["errors"]["max_abs"] for r in group)}
            operators.append({"model": size, "batch": batch, "context": length, "latency": latency,
                "tc_vs_tcsplit_graph_speedup": latency["triton_tc"]["graph_us"] / latency["triton_tc_splitk"]["graph_us"],
                "fp32split_vs_tcsplit_graph_speedup": latency["triton_splitk"]["graph_us"] / latency["triton_tc_splitk"]["graph_us"]})
    match = re.search(r"(\d+) passed in", (base / "tcsplit-all-tests.log").read_text())
    if not match:
        raise ValueError("missing successful full test run")
    return {"tests_passed": int(match[1]), "diagnosis": diagnosis, "models": models,
        "operators": operators, "operator_repeat_records": 150, "e2e_repeat_records": 60,
        "note": "B1 input2048/output128 E2E; operator Graph intervals are separate from profiler busy time. Fixed short prompt bitwise equality is not universal input equivalence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    summary = build_summary(args.results)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", summary)
    lines = ["# TC split-K 实测结果", "", f"完整测试：**{summary['tests_passed']} 项通过**。本次新增150条算子重复记录、60次长输入端到端运行。", "",
        "## 固定短提示回归", "", "| 模型 | 模式 | 测试步数 | logits逐位一致 | token分歧 |", "|---|---|---:|---:|---:|"]
    for model in summary["models"]:
        for row in model["validation"]["cases"]:
            lines.append(f"| {model['model']} | {row['mode']} | {row['tested_steps']} | {row['bitwise_equal_steps']} | {row['argmax_differences']} |")
    lines += ["", "两档使用相同最大上下文2176、三分区容量；短提示只用一个有效分区，验证了空分区和真实Graph路径。", "",
        "## 算子：Graph CUDA-event中位数（μs）", "", "| head几何 | B / KV | FlashAttention | FP32 split-K | TC | TC split-K | TC / TC split-K |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["operators"]:
        latency = row["latency"]
        lines.append(f"| {row['model']} | {row['batch']} / {row['context']} | {latency['flash_attn']['graph_us']:.2f} | {latency['triton_splitk']['graph_us']:.2f} | {latency['triton_tc']['graph_us']:.2f} | {latency['triton_tc_splitk']['graph_us']:.2f} | {row['tc_vs_tcsplit_graph_speedup']:.2f}× |")
    lines += ["", "microbenchmark的small/large分别表示Hq/Hkv为8/2和16/4，D均为256。每项独立重复5次；完整分布、完整调用、峰值显存和独立profiler采样保存在原始记录。事件区间可能包含launch空隙。", "",
        "## 端到端：B=1，输入2048，输出128", "", "| 模型 | 后端 / 模式 | 输出吞吐中位数 tokens/s | 五次min–max | 自由生成与原生一致 |", "|---|---|---:|---:|---|"]
    for model in summary["models"]:
        for row in model["benchmark"]:
            t = row["throughput"]
            lines.append(f"| {model['model']} | {row['backend']} / {row['mode']} | {t['median']:.2f} | {t['min']:.2f}–{t['max']:.2f} | {'是' if row['greedy_matches_native_all_repeats'] else '否'} |")
    lines += ["", "同一模型实例、相同输入ID、相同KV容量；每个full-attention层9页。计时含prefill、decode、采样和结果转换，JIT、capture及预热不计入。linear attention使用官方Torch fallback。自由生成在首次分歧后会使用不同历史，不能将后续token差异视为同输入误差或任务准确率。", "",
        "## 长输入诊断", "", "| 模型 | 目标序列 / step | TC与原生单分区logits一致 | 替换attention输出恢复自动分区 | 同输入样本 |", "|---|---|---|---|---:|"]
    for row in summary["diagnosis"]:
        lines.append(f"| {row['model']} | {row['sequence']} / {row['target_step']} | {row['tc_logits_equal_native_single']} | {row['rescue_logits_equal_auto']} | {row['same_input_samples']} |")
    lines += ["", "算法、因果实验边界与命令见[主报告](../TC_SPLITK_AND_LONG_CONTEXT.md)。", ""]
    if args.plots:
        plot(summary, args.output)
        lines += ["![算子延迟](figures/operator.png)", ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"tests": summary["tests_passed"], "models": len(summary["models"]), "operator_rows": 150, "e2e_rows": 60}))


def plot(summary, output):
    cache = ROOT / ".cache/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = ("flash_attn", "triton_splitk", "triton_tc", "triton_tc_splitk")
    labels = ("FlashAttention", "FP32 split-K", "TC", "TC split-K")
    colors = ("#555555", "#E39936", "#6385BE", "#339A76")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, constrained_layout=True)
    for ax, size in zip(axes, ("small", "large")):
        rows = [r for r in summary["operators"] if r["model"] == size]
        x = np.arange(len(rows))
        for i, (name, label, color) in enumerate(zip(names, labels, colors)):
            vals = [r["latency"][name] for r in rows]
            y = np.array([v["graph_us"] for v in vals])
            errors = np.stack((y - [v["graph_min_us"] for v in vals], [v["graph_max_us"] for v in vals] - y))
            ax.bar(x + (i - 1.5) * .19, y, width=.18, yerr=errors, capsize=2, color=color, label=label)
        ax.set_xticks(x, [f"B={r['batch']}, KV={r['context']}" for r in rows])
        ax.set_title("Hq=8 / Hkv=2" if size == "small" else "Hq=16 / Hkv=4")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Graph event interval (microseconds, log scale)")
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("A100 | D=256 | median and min/max of five runs")
    folder = output / "figures"
    folder.mkdir()
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"operator.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
