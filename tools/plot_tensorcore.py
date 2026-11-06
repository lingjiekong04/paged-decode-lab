"""Render new operator latency samples, preserving min/max across five repeats."""
import argparse
import json
import os
from pathlib import Path
from _common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/TENSORCORE_SUMMARY.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = ROOT / ".cache/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = json.loads(args.input.read_text())["operator_performance"]
    names = ("flash_attn", "triton_base", "triton_splitk", "triton_tc", "tensorcore_bf16")
    labels = ("FlashAttention", "FP32 base", "FP32 split-K", "Ordered TC", "Unordered TC")
    colors = ("#555555", "#4981BA", "#E49933", "#369C76", "#B186C4")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    for ax, size in zip(axes, ("small", "large")):
        selected = [r for r in rows if r["model_geometry"] == size]
        x = np.arange(len(selected))
        for index, (name, label, color) in enumerate(zip(names, labels, colors)):
            values = [r["latency"][name] for r in selected]
            med = np.array([v["graph_event_us"] for v in values])
            lower = med - [v["graph_event_min_us"] for v in values]
            upper = np.array([v["graph_event_max_us"] for v in values]) - med
            ax.bar(x + (index - 2) * .16, med, width=.15, label=label, color=color,
                   yerr=np.stack((lower, upper)), capsize=2, error_kw={"elinewidth": .8})
        ax.set_xticks(x, [f"B={r['batch']}, KV={r['context']}" for r in selected])
        ax.set_title("Hq=8 / Hkv=2" if size == "small" else "Hq=16 / Hkv=4")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Graph event interval (microseconds, log scale)")
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("A100 | D=256 | median and min/max of five runs", fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = args.output / f"operator-latency.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
