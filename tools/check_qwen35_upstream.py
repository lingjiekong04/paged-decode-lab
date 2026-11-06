"""Compare the explicit runner against official HF text forward on identical tokens."""
import argparse
import json
from pathlib import Path
from _common import ROOT, require_gpu, model_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--official-backend", choices=("eager", "flash_attention_2"), default="eager")
    args = parser.parse_args()
    torch = require_gpu()
    from paged_decode.hybrid import HybridRunner
    model_identity(args.model)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    runner = HybridRunner(str(Path(args.model).resolve()), backend="flash_attn", mode="eager", batch=1)
    prompt = json.loads((ROOT / "configs/prompts.json").read_text())[0]
    tokens = runner.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
             tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    custom = runner.generate([tokens], args.tokens, observe=True)
    original = runner.text.config._attn_implementation
    rows = []
    try:
        runner.text.config._attn_implementation = args.official_backend
        cache = None
        inputs = torch.tensor([tokens], device="cuda")
        with torch.inference_mode():
            for step in range(args.tokens):
                outputs = runner.text(input_ids=inputs, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values
                official = runner.model.lm_head(outputs.last_hidden_state[:, -1]).cpu().float()
                actual = custom["logits"][step].float()
                delta = (actual - official).abs()
                finite = bool(torch.isfinite(actual).all() and torch.isfinite(official).all())
                rows.append({"step": step, "max_abs_error": delta.max().item() if finite else None,
                             "mse": delta.square().mean().item() if finite else None,
                             "runner_nonfinite": int((~torch.isfinite(actual)).sum()),
                             "official_nonfinite": int((~torch.isfinite(official)).sum()),
                             "close": finite and bool(torch.allclose(actual, official, atol=2e-2, rtol=2e-2)),
                             "runner_argmax": actual.argmax(-1).tolist(),
                             "official_argmax": official.argmax(-1).tolist()})
                inputs = torch.tensor([[custom["token_ids"][0][step]]], device="cuda")
    finally:
        runner.text.config._attn_implementation = original
    passed = all(r["close"] and r["runner_argmax"] == r["official_argmax"] for r in rows)
    write_json(destination / "summary.json", {"status": "passed" if passed else "numerical_mismatch", "steps": rows,
               "official_backend": args.official_backend,
               "note": "official Transformers text forward, same weights and forced history; NaN/Inf are explicit failures"})
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
