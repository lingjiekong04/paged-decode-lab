"""Check all three backends on identical learned Qwen3.5 decode activations."""
import argparse
import json
from pathlib import Path
from _common import ROOT, require_gpu, model_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=0, help="synthetic fixed prefix; 0 uses the public text prompt")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("step must select a decode step, starting at 1")
    torch = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    from paged_decode.hybrid import HybridRunner
    from paged_decode.runtime import allocate_resources, paged_decode
    from paged_decode.torch_reference import dense_reference
    identity = model_identity(args.model)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    runner = HybridRunner(str(Path(args.model).resolve()), backend="flash_attn", mode="eager", batch=1,
                          max_context=max(2048, args.input_tokens + args.step + 1))
    first = min(runner.cache.pages)
    step = [0]
    reports = []

    def observe(module, q, k, v, native):
        if module.layer_idx == first:
            step[0] += 1
        if step[0] != args.step:
            return
        expected = dense_reference(q, k, v, runner.cache.tables, runner.cache.lengths)
        values = {"flash_attn": native}
        resources = []
        for backend in ("triton_base", "triton_splitk"):
            scratch = allocate_resources(runner.spec, q.device, backend)
            resources.append(scratch)
            values[backend] = paged_decode(q, k, v, runner.cache.tables, runner.cache.lengths,
                                           resources=scratch, backend=backend)
        row = {"layer": module.layer_idx, "step": args.step,
               "context_length": runner.cache.lengths.cpu().tolist(), "backends": {}}
        for backend, value in values.items():
            error = (value.float() - expected).abs()
            row["backends"][backend] = {"max_abs_error": error.max().item(),
                  "mse": error.square().mean().item(),
                  "close": bool(torch.isfinite(value).all() and torch.allclose(value.float(), expected, atol=2e-2, rtol=2e-2)),
                  "max_abs_activation": value.abs().max().item()}
        reports.append(row)

    runner.attention_observer = observe
    prompt = json.loads((ROOT / "configs/prompts.json").read_text())[0]
    tokens = runner.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
             tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    if args.input_tokens:
        if args.input_tokens < 1:
            parser.error("input-tokens must be nonnegative")
        generator = torch.Generator(device="cpu").manual_seed(927)
        tokens = torch.randint(100, 30000, (args.input_tokens,), generator=generator, device="cpu").tolist()
    runner.generate([tokens], args.step + 1)
    if len(reports) != len(runner.cache.pages):
        raise AssertionError("did not observe every full-attention layer")
    passed = all(value["close"] for row in reports for value in row["backends"].values())
    write_json(destination / "summary.json", {"status": "passed" if passed else "failed",
               "model": {k: identity[k] for k in ("repo_id", "revision")}, "layers": reports,
               "partitions": runner.spec.partitions, "input_tokens": len(tokens),
               "note": "identical Q/K/V per layer; does not assert bitwise equality of full-model logits"})
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
