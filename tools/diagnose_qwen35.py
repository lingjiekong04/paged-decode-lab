"""Untimed causal diagnosis; production attention kernels are never modified.

Compare identical teacher-forced histories, layer outputs and SAME-input kernels.
The native-rescue case executes Triton then substitutes native attention before
the model's gate/o_proj. This is an intervention, not a new inference backend.
"""
import argparse
import hashlib
import json
from pathlib import Path

from _common import ROOT, fingerprint, model_identity, require_gpu, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prompt", type=int, required=True)
    parser.add_argument("--step", type=int, required=True, help="zero-based generated logit step")
    parser.add_argument("--output", required=True)
    parser.add_argument("--walkthrough", action="store_true")
    args = parser.parse_args()
    torch = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from flash_attn import flash_attn_with_kvcache
    from transformers import AttentionInterface
    from paged_decode.hybrid import HybridRunner, _attention
    from paged_decode.runtime import allocate_resources, paged_decode
    from paged_decode.torch_reference import dense_reference

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "environment.json", fingerprint())
    identity = model_identity(args.model)
    identity = {k: identity[k] for k in ("repo_id", "revision", "weights", "files")}
    write_json(destination / "arguments.json", {**vars(args), "model": identity,
               "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    runner = HybridRunner(args.model, backend="flash_attn", batch=1, max_context=1024)
    write_json(destination / "runtime.json", runner.description())
    prompts = json.loads((ROOT / "configs/prompts.json").read_text(encoding="utf-8"))
    encoded = runner.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts[args.prompt]}], tokenize=True,
        return_dict=False, add_generation_prompt=True, enable_thinking=False)
    baseline = torch.load(Path(args.reference) / f"prompt-{args.prompt}.pt",
                          map_location="cpu", weights_only=True)
    if baseline["model"] != identity or baseline["prompt_sha256"] != hashlib.sha256(json.dumps(encoded).encode()).hexdigest():
        raise ValueError("model/prompt differs from recorded regression baseline")
    if not 1 <= args.step < len(baseline["tokens"]):
        raise ValueError("target must be a decode step within recorded baseline")

    resources = {name: allocate_resources(runner.spec, runner.device, name)
                 for name in ("triton_base", "triton_splitk")}
    state = {"case": "flash_attn", "step": -1, "walkthrough": False}
    snapshots, same_input, walk = {}, [], []
    selected = {1, args.step - 1, args.step}

    def stats(actual, expected):
        actual, expected = actual.float(), expected.float()
        delta = (actual - expected).abs()
        return {"max_abs": delta.max().item(), "rmse": delta.square().mean().sqrt().item(),
                "bitwise_equal": torch.equal(actual, expected),
                "fraction_unequal": (actual != expected).float().mean().item(),
                "close_2e-2": bool(torch.isfinite(actual).all() and torch.allclose(actual, expected, atol=.02, rtol=.02))}

    def gathered(q, k, v):
        length = int(runner.cache.lengths[0].item())
        pos = torch.arange(length, device=q.device)
        page = runner.cache.tables[0, pos // 256].long()
        return k[page, pos % 256], v[page, pos % 256]

    def native(q, k, v, scale, **kwargs):
        return flash_attn_with_kvcache(q.unsqueeze(1), k, v,
            cache_seqlens=runner.cache.lengths, block_table=runner.cache.tables,
            softmax_scale=scale, causal=True, **kwargs).squeeze(1)

    def diagnostic_attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        output, weights = _attention(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        if runner.cache.is_prefill:
            return output, weights
        q = query.squeeze(2)
        k, v = runner.cache.pages[module.layer_idx]
        step, index = state["step"], module.layer_idx
        if state["walkthrough"]:
            kg, vg = gathered(q, k, v)
            walk.append({"layer": index, "step": step, "q_shape": list(q.shape),
                "q_stride": list(q.stride()), "kv_pool_shape": list(k.shape),
                "lengths": runner.cache.lengths.cpu().tolist(),
                "block_table": runner.cache.tables.cpu().tolist(),
                "new_k_shape": list(key.shape), "output_shape_before_gate": list(output.shape),
                "new_k_matches_logical_last": torch.equal(kg[-1], key[0, :, -1]),
                "new_v_matches_logical_last": torch.equal(vg[-1], value[0, :, -1])})
        elif step in selected:
            kg, vg = gathered(q, k, v)
            snapshots[state["case"]][f"step{step}/attention{index}"] = {
                "q": q.detach().cpu().clone(), "k": kg.cpu().clone(), "v": vg.cpu().clone(),
                "out": output.squeeze(1).detach().cpu().clone()}
            if state["case"] == "flash_attn":
                base = paged_decode(q, k, v, runner.cache.tables, runner.cache.lengths,
                    resources=resources["triton_base"], backend="triton_base", scale=scaling).clone()
                split = paged_decode(q, k, v, runner.cache.tables, runner.cache.lengths,
                    resources=resources["triton_splitk"], backend="triton_splitk", scale=scaling).clone()
                fp32 = dense_reference(q, k, v, runner.cache.tables, runner.cache.lengths)
                group = q.shape[1] // kg.shape[1]
                keys = kg.repeat_interleave(group, dim=1)
                values = vg.repeat_interleave(group, dim=1)
                scores64 = torch.einsum("bhd,nhd->bhn", q.double(), keys.double()) * scaling
                fp64 = torch.einsum("bhn,nhd->bhd", scores64.softmax(-1), values.double())
                scores = torch.einsum("bhd,nhd->bhn", q.float(), keys.float()) * scaling
                p = (scores - scores.amax(-1, keepdim=True)).exp()
                p_bf16 = p.bfloat16().float()
                rounded = torch.einsum("bhn,nhd->bhd", p_bf16, values.float()) / p.sum(-1, keepdim=True)
                native_out = output.squeeze(1)
                row = {"step": step, "layer": index, "length": kg.shape[0],
                       "reference": "FP64 dense softmax then BF16 output", "comparisons_to_fp64_bf16": {},
                       "base_vs_native": stats(base, native_out), "split_vs_base": stats(split, base),
                       "native_num_splits1_vs_auto": stats(native(q, k, v, scaling, num_splits=1), native_out)}
                for name, tensor in (("base", base), ("split", split), ("native", native_out),
                                     ("fp32", fp32.bfloat16()), ("bf16_exp_global", rounded.bfloat16())):
                    row["comparisons_to_fp64_bf16"][name] = stats(tensor.cpu(), fp64.bfloat16().cpu())
                row["bf16_exp_global_vs_native"] = stats(rounded.bfloat16(), native_out)
                same_input.append(row)
        if state["case"] == "native_rescue" and not state["walkthrough"]:
            # Triton already executed above. Replace only attention's pre-gate output.
            output = native(q, k, v, scaling).unsqueeze(1)
        return output, weights

    AttentionInterface.register("paged_decode_lab_diagnosis", diagnostic_attention)
    for layer in runner.text.layers:
        if layer.layer_type == "full_attention":
            layer.self_attn.config._attn_implementation = "paged_decode_lab_diagnosis"

    def layer_hook(index):
        def hook(module, inputs, output):
            if not state["walkthrough"]:
                value = output[0] if isinstance(output, tuple) else output
                snapshots[state["case"]][f"step{state['step']}/layer{index}"] = value.detach().cpu().clone()
        return hook

    handles = [layer.register_forward_hook(layer_hook(index)) for index, layer in enumerate(runner.text.layers)]
    forward = runner._forward
    def traced_forward(tokens, positions):
        state["step"] += 1
        return forward(tokens, positions)
    runner._forward = traced_forward
    cases, result_tensors = {}, {}
    with torch.inference_mode():
        for case in ("flash_attn", "triton_base", "triton_splitk", "native_rescue"):
            print(f"CASE {case}", flush=True)
            state.update(case=case, step=-1)
            snapshots[case] = {}
            runner.backend = "triton_base" if case == "native_rescue" else case
            runner.resources = resources.get(runner.backend)
            result = runner.generate([encoded], len(baseline["tokens"]), observe=True,
                                     forced=[[t] for t in baseline["tokens"]])
            result_tensors[case] = result["logits"]
            reference_logits = result_tensors["flash_attn"]
            rows = []
            for step, (actual, expected) in enumerate(zip(result["logits"], reference_logits)):
                top2 = expected[0].float().topk(2)
                rows.append({"step": step, **stats(actual, expected),
                    "baseline_token": int(expected.argmax(-1).item()),
                    "candidate_token": result["predicted_ids"][0][step],
                    "baseline_top2_gap": float(top2.values[0] - top2.values[1])})
            layers = []
            for step in (0, *sorted(selected)):
                for index in range(len(runner.text.layers)):
                    key = f"step{step}/layer{index}"
                    layers.append({"step": step, "layer": index, "type": runner.text.layers[index].layer_type,
                        **stats(snapshots[case][key], snapshots["flash_attn"][key])})
            attentions = []
            for key, value in snapshots[case].items():
                if "/attention" in key:
                    attentions.append({"location": key, **{name: stats(tensor, snapshots["flash_attn"][key][name])
                                                          for name, tensor in value.items()}})
            report = {"logits": rows, "layers": layers, "attention_inputs_outputs": attentions,
                "argmax_differences": [r["step"] for r in rows if r["baseline_token"] != r["candidate_token"]],
                "all_logits_bitwise_equal": all(r["bitwise_equal"] for r in rows),
                "recorded_baseline_logits": [stats(a, b) for a, b in zip(result["logits"], baseline["logits"])],
                "finite": all(bool(torch.isfinite(x).all()) for x in result["logits"])}
            cases[case] = report
            write_json(destination / f"{case}.json", report)
            # Retain only native snapshots between cases; binary evidence is server-local.
            if case != "flash_attn":
                snapshots[case].clear()
        if args.walkthrough:
            print("TRACE context=257", flush=True)
            state.update(case="flash_attn", step=-1, walkthrough=True)
            runner.backend, runner.resources = "triton_base", resources["triton_base"]
            fixed = (encoded * ((256 + len(encoded) - 1) // len(encoded)))[:256]
            trace_result = runner.generate([fixed], 2, observe=True)
            write_json(destination / "walkthrough-257.json", {
                "prefill_tokens": 256, "decode_position": 256, "context_after_kv_write": 257,
                "runner": runner.description(), "layers": walk,
                "token_ids": trace_result["token_ids"], "finite": True,
                "note": "repeat tokenized prompt to exactly 256 input tokens; tracing workload, not quality benchmark"})
    for handle in handles:
        handle.remove()
    write_json(destination / "same-input.json", same_input)
    torch.save({"native_snapshots": snapshots["flash_attn"], "logits": result_tensors}, destination / "tensors.pt")
    write_json(destination / "summary.json", {"status": "diagnosis_completed", "prompt": args.prompt,
        "prompt_tokens": len(encoded), "target_step": args.step, "model": identity,
        "cases": {k: {f: v[f] for f in ("argmax_differences", "all_logits_bitwise_equal", "finite")}
                  for k, v in cases.items()}, "same_input_samples": len(same_input),
        "production_source_modified": False, "strict_regression_status": "still numerical_mismatch"})
    print(json.dumps({k: v["argmax_differences"] for k, v in cases.items()}), flush=True)


if __name__ == "__main__":
    main()
