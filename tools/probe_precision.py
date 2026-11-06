"""Evaluate experimental BF16-exp kernels without changing production backends."""
import argparse
import hashlib
import json
from pathlib import Path

from _common import ROOT, fingerprint, model_identity, require_gpu, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("activations", "model"), required=True)
    parser.add_argument("--input", help="server-local diagnosis tensors.pt")
    parser.add_argument("--model")
    parser.add_argument("--reference", help="recorded flash_attn-eager baseline directory")
    parser.add_argument("--variants", nargs="+", default=["scalar_bf16", "tensorcore_bf16"])
    parser.add_argument("--modes", nargs="+", choices=("eager", "graph"), default=["eager", "graph"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch = require_gpu()
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    from paged_decode.experimental_precision import VARIANTS, precision_decode
    from paged_decode.runtime import allocate_resources
    from paged_decode.spec import DecodeSpec
    public_backends = ("triton_tc", "triton_tc_splitk")
    if any(v not in (*VARIANTS, *public_backends) for v in args.variants):
        parser.error("unknown precision variant")
    if args.task == "activations" and any(v in public_backends for v in args.variants):
        parser.error("use tensorcore_ordered for same-input activation audit")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "environment.json", fingerprint())
    write_json(destination / "arguments.json", vars(args))

    def metrics(actual, expected):
        a, e = actual.float().cpu(), expected.float().cpu()
        delta = a - e
        return {"max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
                "fraction_unequal": (a != e).float().mean().item(), "bitwise_equal": torch.equal(a, e),
                "close_2e-2": bool(torch.isfinite(a).all() and torch.allclose(a, e, atol=.02, rtol=.02))}

    if args.task == "activations":
        if not args.input:
            parser.error("activations requires --input")
        inputs = torch.load(args.input, map_location="cpu", weights_only=True)["native_snapshots"]
        rows = []
        for location, values in inputs.items():
            if "/attention" not in location:
                continue
            q, keys, vals, expected = (values[k].cuda() for k in ("q", "k", "v", "out"))
            length, hkv, d = keys.shape
            spec = DecodeSpec(1, q.shape[1], hkv, d, max(256, length))
            width = spec.table_width
            table = torch.randperm(width, device="cuda", dtype=torch.int64).to(torch.int32).view(1, width)
            k, v = (torch.full((width, 256, hkv, d), float("nan"), device="cuda", dtype=torch.bfloat16) for _ in range(2))
            positions = torch.arange(length, device="cuda")
            pages = table[0, positions // 256].long()
            k[pages, positions % 256], v[pages, positions % 256] = keys, vals
            lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
            resources = allocate_resources(spec, "cuda", "triton_base")
            for variant in args.variants:
                out = precision_decode(q, k, v, table, lengths, resources=resources, variant=variant).clone()
                rows.append({"location": location, "length": length, "variant": variant, **metrics(out, expected)})
        write_json(destination / "summary.json", {"status": "completed", "rows": rows,
            "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest()})
        for variant in args.variants:
            selected = [r for r in rows if r["variant"] == variant]
            print(variant, "exact", sum(r["bitwise_equal"] for r in selected), "/", len(selected),
                  "worst", max(r["max_abs"] for r in selected), flush=True)
        return

    if not args.model or not args.reference:
        parser.error("model task requires --model and --reference")
    from transformers import AttentionInterface
    from paged_decode.hybrid import HybridRunner, _attention
    runner = HybridRunner(args.model, backend="triton_base", mode="eager", batch=1, max_context=1024)
    identity = model_identity(args.model)
    identity = {k: identity[k] for k in ("repo_id", "revision", "weights", "files")}
    write_json(destination / "model.json", identity)
    selected = {"variant": args.variants[0]}

    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        if runner.cache.is_prefill:
            return _attention(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        if attention_mask is not None or dropout or scaling != runner.spec.head_dim ** -.5:
            raise ValueError("unsupported attention options in precision probe")
        k, v = runner.cache.pages[module.layer_idx]
        output = precision_decode(query.squeeze(2), k, v, runner.cache.tables, runner.cache.lengths,
                                 resources=runner.resources, variant=selected["variant"])
        return output.unsqueeze(1), None

    AttentionInterface.register("paged_decode_lab_precision_probe", attention)
    for layer in runner.text.layers:
        if layer.layer_type == "full_attention":
            layer.self_attn.config._attn_implementation = "paged_decode_lab_precision_probe"
    prompts = json.loads((ROOT / "configs/prompts.json").read_text(encoding="utf-8"))
    all_reports = []
    for variant in args.variants:
        selected["variant"] = variant
        runner.backend = variant if variant in public_backends else "triton_base"
        runner.resources = allocate_resources(runner.spec, runner.device, runner.backend)
        for layer in runner.text.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.config._attn_implementation = ("paged_decode_lab_qwen35" if variant in public_backends
                                                               else "paged_decode_lab_precision_probe")
        for mode in args.modes:
            print("CASE", variant, mode, flush=True)
            runner.graph = None
            runner.mode = mode
            if mode == "graph":
                runner._capture()
            case = destination / f"{variant}-{mode}"
            case.mkdir()
            reports = []
            for index, prompt in enumerate(prompts):
                encoded = runner.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                baseline = torch.load(Path(args.reference) / f"prompt-{index}.pt", map_location="cpu", weights_only=True)
                if baseline["model"] != identity or baseline["prompt_sha256"] != hashlib.sha256(json.dumps(encoded).encode()).hexdigest():
                    raise ValueError("model or prompt differs from baseline")
                result = runner.generate([encoded], len(baseline["tokens"]), observe=True,
                                         forced=[[t] for t in baseline["tokens"]])
                rows = []
                for step, (actual, expected) in enumerate(zip(result["logits"], baseline["logits"])):
                    rows.append({"step": step, **metrics(actual, expected),
                        "baseline_token": baseline["tokens"][step],
                        "candidate_token": result["predicted_ids"][0][step]})
                differences = [r["step"] for r in rows if r["baseline_token"] != r["candidate_token"]]
                report = {"prompt": index, "steps": rows, "argmax_differences": differences,
                          "passed": not differences and all(r["close_2e-2"] for r in rows)}
                write_json(case / f"prompt-{index}.json", report)
                reports.append(report)
            summary = {"variant": variant, "mode": mode, "prompts": reports,
                "argmax_differences": sum(len(r["argmax_differences"]) for r in reports),
                "logits_failed_steps": sum(not s["close_2e-2"] for r in reports for s in r["steps"]),
                "passed": all(r["passed"] for r in reports)}
            write_json(case / "summary.json", summary)
            all_reports.append({k: v for k, v in summary.items() if k != "prompts"})
            print(json.dumps(all_reports[-1]), flush=True)
    passed = all(r["passed"] for r in all_reports)
    write_json(destination / "summary.json", {"status": "passed" if passed else "numerical_mismatch",
                                             "cases": all_reports, "production_backend": all(v in public_backends for v in args.variants)})
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
