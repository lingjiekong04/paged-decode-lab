"""Summarize completed diagnosis JSON without importing Torch or CUDA."""
import argparse
import json
import math
from pathlib import Path

from _common import write_json


def summarize(directory):
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))
    summary = read("summary.json")
    samples = read("same-input.json")
    base, native, rescue = (read(f"{name}.json") for name in ("triton_base", "flash_attn", "native_rescue"))
    rms = lambda values: math.sqrt(sum(x * x for x in values) / len(values))
    base_native = rms([r["base_vs_native"]["rmse"] for r in samples])
    rounded_native = rms([r["bf16_exp_global_vs_native"]["rmse"] for r in samples])
    target = summary["target_step"]
    first = next((r for r in base["layers"] if r["step"] == 1 and not r["bitwise_equal"]), None)
    return {"model": summary["model"]["repo_id"], "prompt": summary["prompt"],
        "prompt_tokens": summary["prompt_tokens"], "target_step": target,
        "same_input_samples": len(samples), "same_input_lengths": sorted({r["length"] for r in samples}),
        "first_different_layer_at_step1": first,
        "target_logits": base["logits"][target],
        "native_reproduces_recorded_baseline_bitwise": all(r["bitwise_equal"] for r in native["recorded_baseline_logits"]),
        "native_rescue_logits_bitwise": rescue["all_logits_bitwise_equal"],
        "native_rescue_sampled_layer_outputs_bitwise": all(r["bitwise_equal"] for r in rescue["layers"]),
        "first_full_layer_qkv_identical": all(all(r[k]["bitwise_equal"] for k in ("q", "k", "v"))
            for r in base["attention_inputs_outputs"] if r["location"].endswith("/attention3")),
        "native_num_splits1_equals_auto": all(r["native_num_splits1_vs_auto"]["bitwise_equal"] for r in samples),
        "base_and_split_same_input_bitwise": all(r["split_vs_base"]["bitwise_equal"] for r in samples),
        "pooled_rmse_to_fp64_then_bf16": {k: rms([r["comparisons_to_fp64_bf16"][k]["rmse"] for r in samples])
                                          for k in ("base", "split", "native", "fp32", "bf16_exp_global")},
        "base_native_pooled_rmse": base_native,
        "bf16_exp_native_pooled_rmse": rounded_native,
        "rmse_reduction_after_bf16_exp_percent": (1 - rounded_native / base_native) * 100 if base_native else None,
        "bf16_exp_native_exact_samples": sum(r["bf16_exp_global_vs_native"]["bitwise_equal"] for r in samples),
        "bf16_exp_native_unequal_elements_percent": 100 * sum(r["bf16_exp_global_vs_native"]["fraction_unequal"] for r in samples) / len(samples),
        "base_fp64_exact_samples": sum(r["comparisons_to_fp64_bf16"]["base"]["bitwise_equal"] for r in samples),
        "base_fp64_worst_abs": max(r["comparisons_to_fp64_bf16"]["base"]["max_abs"] for r in samples),
        "all_same_input_base_pass_original_tolerance": all(r["comparisons_to_fp64_bf16"]["base"]["close_2e-2"] for r in samples),
        "note": "Pooled RMS weights every selected layer/step equally; head geometry is constant within one model. No performance claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [summarize(p) for p in args.input]
    write_json(args.output, rows)
    for row in rows:
        print(row["model"], "native rescue:", row["native_rescue_logits_bitwise"],
              "BF16-P RMSE reduction:", row["rmse_reduction_after_bf16_exp_percent"])


if __name__ == "__main__":
    main()
