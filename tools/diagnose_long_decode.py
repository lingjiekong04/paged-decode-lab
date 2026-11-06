"""Reproduce the first E2E divergence with identical histories and layer traces.

Compare native auto, native single partition, TC, and TC with native-auto output
substitution. Audit the exact Q/K/V from the native-auto path. Untimed eager only.
"""
import argparse
import hashlib
import json
from pathlib import Path
from _common import fingerprint, model_identity, require_gpu, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--e2e", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-backend", choices=("triton_tc", "triton_tc_splitk"), default="triton_tc")
    parser.add_argument("--full-history", action="store_true", help="trace the full recorded output, beyond the first difference")
    parser.add_argument("--matched", action="store_true", help="also test the experimental capacity/LSE-aligned path and its Graph")
    args = parser.parse_args()
    torch = require_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from flash_attn import flash_attn_with_kvcache
    from transformers import AttentionInterface
    from paged_decode.hybrid import HybridRunner, _attention
    from paged_decode.runtime import allocate_resources, paged_decode
    from paged_decode.torch_reference import dense_reference
    args.output.mkdir(parents=True, exist_ok=False)
    read = lambda name: json.loads((args.e2e / name).read_text())
    settings = read("arguments.json")
    modern = (args.e2e / "bench-flash_attn-eager-r0.json").exists()
    original = read("bench-flash_attn-eager-r0.json" if modern else "flash_attn-eager-r0.json")
    candidate = read(("bench-" if modern else "") + args.candidate_backend + "-eager-r0.json")
    if modern:
        original["token_ids"] = [original["token_ids"]]
        candidate["token_ids"] = [candidate["token_ids"]]
    settings["batch"] = settings.get("batch", 1)
    identity = model_identity(args.model)
    identity_keys = ("repo_id", "revision", "weights", "files")
    if {k: settings["model_identity"][k] for k in identity_keys} != {k: identity[k] for k in identity_keys}:
        raise ValueError("model identity differs from E2E run")
    differences = [(step, b) for b, (left, right) in enumerate(zip(original["token_ids"], candidate["token_ids"]))
                   for step, (a, c) in enumerate(zip(left, right)) if a != c]
    if not differences:
        raise ValueError("no E2E divergence to investigate")
    target, sequence = min(differences)
    steps = settings["tokens"] if args.full_history else target + 1
    generator = torch.Generator(device="cpu").manual_seed(927)
    prompts = torch.randint(100, 30000, (settings["batch"], settings["input_tokens"]),
                           generator=generator, device="cpu").tolist()
    if hashlib.sha256(json.dumps(prompts).encode()).hexdigest() != original["prompt_sha256"]:
        raise ValueError("input token IDs changed")
    forced = [[tokens[step] for tokens in original["token_ids"]] for step in range(steps)]
    write_json(args.output / "environment.json", fingerprint())
    write_json(args.output / "arguments.json", {"model": identity, "e2e": str(args.e2e),
        "target_step": target, "target_sequence": sequence, "input_tokens": settings["input_tokens"],
        "candidate_backend": args.candidate_backend, "full_history": args.full_history, "matched": args.matched,
        "prompt_sha256": original["prompt_sha256"], "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    runner = HybridRunner(args.model, backend="flash_attn", batch=settings["batch"],
                          max_context=max(1024, settings["input_tokens"] + settings["tokens"]))
    resources = allocate_resources(runner.spec, runner.device, args.candidate_backend)
    aligned = None
    if args.matched:
        from dataclasses import asdict
        from paged_decode.matched_splitkv import allocate_matched_resources, matched_decode
        aligned = allocate_matched_resources(runner.spec, runner.device)
        write_json(args.output / "matched-plan.json", asdict(aligned.plan))
    state = {"case": "native_auto", "step": -1}
    selected = {1, max(1, target - 1), target}
    snapshots, audits, logits, reports = {}, [], {}, {}

    def metric(actual, expected):
        a, b = actual.float(), expected.float()
        delta = a - b
        return {"max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
            "bitwise_equal": torch.equal(actual, expected),
            "fraction_unequal": (a != b).float().mean().item(),
            "close_2e-2": bool(torch.isfinite(a).all() and torch.allclose(a, b, atol=.02, rtol=.02))}

    def native(q, k, v, scale, splits=0):
        return flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=runner.cache.lengths,
            block_table=runner.cache.tables, causal=True, softmax_scale=scale, num_splits=splits).squeeze(1)

    def gather(k, v):
        length = int(runner.cache.lengths[0].item())
        pos = torch.arange(length, device=runner.device)
        pages = runner.cache.tables[:, pos // 256].long()
        return k[pages, pos[None, :] % 256], v[pages, pos[None, :] % 256]

    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        if runner.cache.is_prefill:
            return _attention(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        q = query.squeeze(2)
        k, v = runner.cache.pages[module.layer_idx]
        if state["case"] == "native_single":
            computed = native(q, k, v, scaling, 1)
        elif state["case"] == "matched":
            computed = matched_decode(q,k,v,runner.cache.tables,runner.cache.lengths,resources=aligned,scale=scaling)
        else:
            computed = _attention(module, query, key, value, attention_mask, scaling, dropout, **kwargs)[0].squeeze(1)
        output = native(q, k, v, scaling) if state["case"] == "tc_rescue_auto" else computed
        if state["step"] in selected:
            kg, vg = gather(k, v)
            location = f"step{state['step']}/attention{module.layer_idx}"
            snapshots[state["case"]][location] = {"q": q.cpu().clone(), "k": kg.cpu().clone(),
                "v": vg.cpu().clone(), "computed_out": computed.cpu().clone(), "returned_out": output.cpu().clone()}
            if state["case"] == "native_auto":
                tc = paged_decode(q, k, v, runner.cache.tables, runner.cache.lengths,
                                  resources=resources, backend=args.candidate_backend, scale=scaling).clone()
                fixed = {n: native(q, k, v, scaling, n) for n in (1, 2, 4, 8, 16)}
                gold = dense_reference(q, k, v, runner.cache.tables, runner.cache.lengths)
                audits.append({"location": location, "context": kg.shape[1],
                    "tc_vs_auto": metric(tc, output), "tc_vs_native_single": metric(tc, fixed[1]),
                    "auto_vs_fixed": {str(n): metric(output, tensor) for n, tensor in fixed.items()},
                    "matching_fixed_splits": [n for n, tensor in fixed.items() if torch.equal(tensor, output)],
                    "tc_vs_fp32": metric(tc, gold), "auto_vs_fp32": metric(output, gold)})
                if aligned is not None:
                    matched = matched_decode(q,k,v,runner.cache.tables,runner.cache.lengths,resources=aligned,scale=scaling).clone()
                    audits[-1]["matched_vs_auto"] = metric(matched,output)
                    audits[-1]["predicted_fixed_vs_auto"] = metric(native(q,k,v,scaling,aligned.plan.splits),output)
        return output.unsqueeze(1), None

    AttentionInterface.register("paged_decode_lab_long_diagnosis", attention)
    for layer in runner.text.layers:
        if layer.layer_type == "full_attention":
            layer.self_attn.config._attn_implementation = "paged_decode_lab_long_diagnosis"

    def hook(index):
        def record(module, inputs, output):
            if state["step"] == 0 or state["step"] in selected:
                snapshots[state["case"]][f"step{state['step']}/layer{index}"] = output[:, -1:].cpu().clone()
        return record
    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(runner.text.layers)]
    forward = runner._forward
    def tracked(tokens, positions):
        state["step"] += 1
        return forward(tokens, positions)
    runner._forward = tracked
    with torch.inference_mode():
        for case in ("native_auto", "native_single", "tc", "tc_rescue_auto") + (("matched",) if args.matched else ()):
            print("CASE", case, flush=True)
            state.update(case=case, step=-1)
            snapshots[case] = {}
            runner.backend = args.candidate_backend if case.startswith("tc") else "flash_attn"
            runner.resources = resources if case.startswith("tc") else None
            result = runner.generate(prompts, steps, forced=forced, observe=True)
            rows = []
            for step, tensor in enumerate(result["logits"]):
                row = {"step": step, "vs_native_auto": metric(tensor, logits["native_auto"][step]) if case != "native_auto" else metric(tensor, tensor),
                    "baseline_ids": forced[step], "predicted_ids": [r[step] for r in result["predicted_ids"]]}
                if "native_single" in logits:
                    row["vs_native_single"] = metric(tensor, logits["native_single"][step])
                rows.append(row)
            if case in ("native_auto", "native_single"):
                logits[case] = result["logits"]
            layers, inputs = [], []
            for name, tensor in snapshots[case].items():
                expected = snapshots["native_auto"][name]
                if "/layer" in name:
                    layers.append({"location": name, **metric(tensor, expected)})
                else:
                    inputs.append({"location": name, **{key: metric(tensor[key], expected[key]) for key in tensor}})
            report = {"case": case, "steps": rows, "layers": layers, "attention_inputs_outputs": inputs,
                "all_logits_equal_auto": all(r["vs_native_auto"]["bitwise_equal"] for r in rows),
                "argmax_differences": [{"step": r["step"], "sequence": b, "baseline": a, "actual": c}
                    for r in rows for b, (a, c) in enumerate(zip(r["baseline_ids"], r["predicted_ids"])) if a != c]}
            if case == "native_auto" and report["argmax_differences"]:
                raise RuntimeError("native baseline no longer reproduces E2E history")
            write_json(args.output / f"{case}.json", report)
            reports[case] = {k: report[k] for k in ("all_logits_equal_auto", "argmax_differences")}
            if case != "native_auto":
                snapshots[case].clear()
    for handle in handles:
        handle.remove()
    runner._forward = forward
    if aligned is not None:
        def graph_attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
            if runner.cache.is_prefill:
                return _attention(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
            k,v = runner.cache.pages[module.layer_idx]
            out = matched_decode(query.squeeze(2),k,v,runner.cache.tables,runner.cache.lengths,resources=aligned,scale=scaling)
            return out.unsqueeze(1),None
        AttentionInterface.register("paged_decode_lab_long_matched",graph_attention)
        for layer in runner.text.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.config._attn_implementation = "paged_decode_lab_long_matched"
        runner.backend = "flash_attn"
        runner.mode = "graph"
        runner._capture()
        result = runner.generate(prompts,steps,forced=forced,observe=True)
        rows = [{"step":step, **metric(tensor,logits["native_auto"][step])}
                for step,tensor in enumerate(result["logits"])]
        reports["matched_graph"] = {"all_logits_equal_auto":all(r["bitwise_equal"] for r in rows),
            "argmax_differences":[{"step":s,"sequence":b,"baseline":a,"actual":c}
                for b,tokens in enumerate(result["predicted_ids"]) for s,(a,c) in enumerate(zip(original["token_ids"][b][:steps],tokens)) if a!=c]}
        write_json(args.output/"matched_graph.json",{"steps":rows,**reports["matched_graph"]})
        free = runner.generate(prompts,steps)
        write_json(args.output/"matched_free_graph.json",{"token_ids":free["token_ids"],
            "matches_native":free["token_ids"]==[r[:steps] for r in original["token_ids"]]})
    write_json(args.output / "same-input.json", audits)
    binary = args.output / "native-activations.pt"
    torch.save({"snapshots": snapshots["native_auto"], "forced_tokens": forced,
                "target_logits": {name: values[target] for name, values in logits.items()}}, binary)
    write_json(args.output / "summary.json", {"status": "diagnosis_completed", "target_step": target,
        "target_sequence": sequence, "batch": settings["batch"], "cases": reports,
        "traced_steps":steps,
        "same_input_samples": len(audits), "binary": {"name": binary.name, "bytes": binary.stat().st_size,
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
        "note": "matching fixed splits inferred only from identical outputs; actual native heuristic choice is not instrumented"})
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
