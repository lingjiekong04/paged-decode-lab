"""Plot measured layer-error propagation and same-input precision ablation."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = Path(__file__).resolve().parents[1] / ".cache/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from summarize_diagnosis import summarize
    report = summarize(args.input)
    base = json.loads((args.input / "triton_base.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for step in (1, report["target_step"]):
        rows = [r for r in base["layers"] if r["step"] == step]
        axes[0].plot([r["layer"] for r in rows], [r["rmse"] for r in rows], marker=".", label=f"step {step}")
    axes[0].set(xlabel="Decoder layer (zero based)", ylabel="Hidden output RMSE vs native",
                title="Error starts at first full attention")
    axes[0].axvline(3, color="gray", linestyle="--", alpha=.6)
    axes[0].legend()
    values = report["pooled_rmse_to_fp64_then_bf16"]
    names = ["base", "native", "bf16_exp_global"]
    axes[1].bar(["Triton base", "FlashAttention", "Dense + BF16 exp"],
                [values[k] for k in names], color=["#2272B6", "#E68632", "#64A65A"])
    axes[1].set(yscale="log", ylabel="Pooled RMSE vs FP64 then BF16",
                title=f"Same Q/K/V: {report['same_input_samples']} samples")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle(report["model"] + " | teacher-forced eager diagnosis", fontsize=12)
    for suffix in ("png", "svg"):
        target = args.output / f"numerical-diagnosis.{suffix}"
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
