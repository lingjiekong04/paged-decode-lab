"""Untimed layer-level comparison on the SAME real model Q/K/V activations."""
import argparse
import json
from pathlib import Path
from _common import ROOT, require_gpu, model_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prompt", type=int, default=1)
    parser.add_argument("--step", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.attention import Attention
    from nanovllm.utils.context import get_context
    from paged_decode.spec import DecodeSpec
    from paged_decode.runtime import allocate_resources, paged_decode
    from paged_decode.torch_reference import dense_reference
    model_identity(args.model)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    baseline = torch.load(Path(args.reference) / f"prompt-{args.prompt}.pt", weights_only=True, map_location="cpu")
    if not 1 <= args.step < len(baseline["tokens"]):
        raise ValueError("step must select an existing decode step (prefill is step zero)")
    prompts = json.loads((ROOT / "configs/prompts.json").read_text(encoding="utf-8"))
    llm = None
    hooks = []
    reports = []
    counter = [0]
    token_cursor = [0]
    try:
        llm = LLM(str(Path(args.model).resolve()), max_num_seqs=8, max_model_len=1024,
                  max_num_batched_tokens=4096, kv_cache_blocks=64,
                  decode_backend="flash_attn", enforce_eager=True, gpu_memory_utilization=.75)
        encoded = llm.tokenizer.apply_chat_template([{"role": "user", "content": prompts[args.prompt]}],
                  tokenize=True, add_generation_prompt=True, enable_thinking=False)

        def observe_layer(name):
            def observe(module, inputs, output):
                context = get_context()
                if context.is_prefill or counter[0] != args.step:
                    return
                q = inputs[0]
                k, v = module.k_cache, module.v_cache
                tables, lengths = context.block_tables, context.context_lens
                expected = dense_reference(q, k, v, tables, lengths)
                shape = DecodeSpec(q.shape[0], q.shape[1], k.shape[2], q.shape[2], 1024)
                values = {"flash_attn": output.squeeze(1)}
                resources = []
                for backend in ("triton_base", "triton_splitk"):
                    resource = allocate_resources(shape, q.device, backend)
                    resources.append(resource)
                    values[backend] = paged_decode(q, k, v, tables, lengths, resources=resource, backend=backend)
                row = {"layer": name, "step": args.step, "context_length": lengths.cpu().tolist(), "backends": {}}
                for backend, value in values.items():
                    delta = (value.float() - expected).abs()
                    row["backends"][backend] = {
                        "max_abs": delta.max().item(), "mse": delta.square().mean().item(),
                        "close": bool(torch.allclose(value.float(), expected, atol=2e-2, rtol=2e-2)),
                        "max_abs_activation": value.abs().max().item()}
                reports.append(row)
            return observe

        for name, module in llm.model_runner.model.named_modules():
            if isinstance(module, Attention):
                hooks.append(module.register_forward_hook(observe_layer(name)))

        def advance(module, inputs, output):
            counter[0] += 1

        def force(module, inputs, output):
            token = baseline["tokens"][token_cursor[0]]
            token_cursor[0] += 1
            return torch.full_like(output, token)

        hooks.append(llm.model_runner.model.lm_head.register_forward_hook(advance))
        hooks.append(llm.model_runner.sampler.register_forward_hook(force))
        llm.generate([encoded], SamplingParams(temperature=0, max_tokens=args.step + 1, ignore_eos=True), use_tqdm=False)
        if len(reports) != llm.model_runner.config.hf_config.num_hidden_layers:
            raise AssertionError("did not observe every attention layer")
        passed = all(b["close"] for r in reports for b in r["backends"].values())
        write_json(destination / "summary.json", {"status": "passed" if passed else "failed", "layers": reports,
                   "note": "same baseline activations; checks per-layer attention, not full-model bitwise equivalence"})
        if not passed:
            raise SystemExit(1)
    finally:
        for hook in hooks:
            hook.remove()
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
