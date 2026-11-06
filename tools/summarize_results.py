"""Generate tables/plots only from completed real benchmark outputs."""
import argparse
import csv
import json
from pathlib import Path
from _common import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    operator = json.loads((Path(args.operators) / "summary.json").read_text())
    suite = json.loads((Path(args.engine) / "suite.json").read_text())
    if operator["status"] != "completed" or suite["status"] != "completed":
        raise ValueError("only completed campaigns may be summarized")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    groups = {}
    for row in operator["rows"]:
        key = (row["batch"], row["context"], row["head_dim"], row["backend"])
        groups.setdefault(key, []).append(row)
    op_rows = []
    for (batch, context, dim, backend), rows in sorted(groups.items()):
        op_rows.append({"batch": batch, "context": context, "head_dim": dim, "backend": backend,
                        "event_us_median": summarize([r["preallocated_event_us"] for r in rows])["median"],
                        "graph_us_median": summarize([r["graph_event_us"] for r in rows])["median"],
                        "full_wall_us_median": summarize([s for r in rows for s in r["full_call_wall_us"]])["median"],
                        "repeats": len(rows)})
    engine_rows = []
    identity = None
    run_identity = None
    workload = None
    model_name = "Model"
    for run in suite["runs"]:
        run_root = Path(args.engine) / run["name"]
        summary = json.loads((run_root / "summary.json").read_text())
        configuration = json.loads((run_root / "arguments.json").read_text())
        model = configuration.get("model_identity", configuration.get("model"))
        if not isinstance(model, dict):
            raise ValueError("missing model identity in benchmark arguments")
        current_run = (model["repo_id"], model["revision"], configuration["batch"],
                       configuration["input_tokens"], configuration.get("tokens", configuration.get("output_tokens")))
        if run_identity is not None and current_run != run_identity:
            raise ValueError("model identity or workload differs between configurations")
        run_identity = current_run
        model_name = model["repo_id"].split("/")[-1]
        workload = current_run[2:]
        if summary["status"] != "completed":
            raise ValueError("incomplete engine run")
        rows = summary["rows"]
        current = {(r["prompt_sha256"], r.get("kv_blocks", r.get("pages_per_full_layer")), r["output_tokens"]) for r in rows}
        if len(current) != 1 or identity is not None and current != identity:
            raise ValueError("engine inputs, output counts or KV budgets differ")
        identity = current
        engine_rows.append({"backend": rows[0]["backend"], "mode": rows[0]["mode"],
                            "tokens_s_median": summary["throughput"]["median"],
                            "tokens_s_min": summary["throughput"]["min"],
                            "tokens_s_max": summary["throughput"]["max"],
                            "decode_model_ms_median": summarize([t for r in rows for t in r.get(
                                "decode_model_ms_raw", [i["milliseconds"] for i in r.get("decode_model_intervals", [])])])["median"],
                            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
                            "repeats": len(rows)})
    for filename, rows in (("operator.csv", op_rows), ("engine.csv", engine_rows)):
        with (destination / filename).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = [f"# Measured results: {model_name}", "",
              f"Engine workload: batch={workload[0]}, input={workload[1]}, output={workload[2]} tokens per sequence. Median across repeats; raw samples retained.", "",
              "CUDA-event intervals include possible launch gaps. Graph replay reduces host launch overhead but is still not pure kernel busy time. Full-call wall time includes synchronization and allocation.", "", "## Operator", "", "| B | KV length | D | backend | eager event μs | graph event μs | full wall μs |", "|---:|---:|---:|---|---:|---:|---:|"]
    for r in op_rows:
        report.append(f"| {r['batch']} | {r['context']} | {r['head_dim']} | {r['backend']} | {r['event_us_median']:.3f} | {r['graph_us_median']:.3f} | {r['full_wall_us_median']:.3f} |")
    report += ["", "## End-to-end", "", "| backend | mode | output tokens/s | decode model ms median | peak allocated GiB |", "|---|---|---:|---:|---:|"]
    for r in engine_rows:
        report.append(f"| {r['backend']} | {r['mode']} | {r['tokens_s_median']:.2f} | {r['decode_model_ms_median']:.3f} | {r['peak_allocated_bytes']/2**30:.3f} |")
    report += ["", "These are fixed-workload measurements, not maximum serving capacity or production guarantees.", ""]
    (destination / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        labels = [r["backend"] + "\n" + r["mode"] for r in engine_rows]
        medians = [r["tokens_s_median"] for r in engine_rows]
        ranges = [[r["tokens_s_median"] - r["tokens_s_min"] for r in engine_rows],
                  [r["tokens_s_max"] - r["tokens_s_median"] for r in engine_rows]]
        ax.bar(labels, medians, yerr=ranges, capsize=4)
        ax.set_ylabel("Output tokens/s (generation wall time)")
        ax.set_title(f"{model_name}: B={workload[0]}, input={workload[1]}, output={workload[2]}\n"
                     "Median throughput; whiskers show min–max across repeats")
        fig.savefig(destination / "engine-throughput.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        plot_dim = max(r["head_dim"] for r in op_rows)
        for ax, batch in zip(axes, (1, 32)):
            for backend in ("flash_attn", "triton_base", "triton_splitk"):
                rows = sorted((r for r in op_rows if r["batch"] == batch and r["head_dim"] == plot_dim
                               and r["backend"] == backend), key=lambda r: r["context"])
                ax.plot([r["context"] for r in rows], [r["graph_us_median"] for r in rows], marker="o", label=backend)
            ax.set_title(f"Batch {batch}, head dimension {plot_dim}")
            ax.set_xlabel("KV length")
            ax.set_ylabel("Graph CUDA-event interval (μs)")
            ax.set_yscale("log")
            ax.legend()
        fig.savefig(destination / "operator-latency.png", dpi=180)
        plt.close(fig)
    print(destination / "RESULTS.md")


if __name__ == "__main__":
    main()
