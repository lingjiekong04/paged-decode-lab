"""Text-only Qwen3.5 fixed-batch runner using official Transformers 5.3 blocks.

Full attention uses our paged pool, linear attention uses the official Gated
DeltaNet blocks. Stable state copies preserve addresses for full decode graphs.
This adapter is separate from the pinned nano-vLLM Qwen3 scheduler.
"""
import torch
from transformers import AttentionInterface, AutoTokenizer, Qwen3_5ForConditionalGeneration

from .runtime import allocate_resources, paged_decode
from .spec import DecodeSpec
from .hybrid_options import validate_hybrid_options


def allocate_hybrid_resources(spec, device, backend):
    if backend == "flash_attn":
        return None
    if backend == "triton_matched":
        from .matched_splitkv import allocate_matched_resources
        return allocate_matched_resources(spec, device)
    return allocate_resources(spec, device, backend)


class StableStates(list):
    """Keep recurrent/conv state addresses stable across steps and Graph replay."""
    def __setitem__(self, index, value):
        previous = self[index]
        if previous is None:
            super().__setitem__(index, value.detach().clone())
        else:
            if previous.shape != value.shape or previous.dtype != value.dtype:
                raise ValueError("hybrid state geometry changed; create a new fixed-batch session")
            previous.copy_(value)


class PagedHybridCache:
    def __init__(self, config, batch, maximum, device):
        self.batch = batch
        self.maximum = maximum
        self.width = (maximum + 255) // 256
        self.prefilled = False
        self.is_prefill = True
        self.conv_states = StableStates([None] * config.num_hidden_layers)
        self.recurrent_states = StableStates([None] * config.num_hidden_layers)
        generator = torch.Generator(device="cpu").manual_seed(2026)
        self.tables = torch.randperm(batch * self.width, generator=generator,
                                     device="cpu", dtype=torch.int64).to(device=device, dtype=torch.int32).view(batch, self.width)
        self.lengths = torch.zeros(batch, device=device, dtype=torch.int32)
        shape = (batch * self.width, 256, config.num_key_value_heads, config.head_dim)
        self.pages = {index: (torch.empty(shape, device=device, dtype=torch.bfloat16),
                              torch.empty(shape, device=device, dtype=torch.bfloat16))
                      for index, kind in enumerate(config.layer_types) if kind == "full_attention"}

    @property
    def has_previous_state(self):
        return self.prefilled

    def reset(self):
        self.prefilled = False
        self.is_prefill = True
        self.lengths.zero_()

    def update(self, key, value, layer_idx, cache_kwargs):
        position = cache_kwargs["cache_position"]
        physical = self.tables[:, position // 256].long() * 256 + position[None, :] % 256
        k, v = self.pages[layer_idx]
        hkv, dim = k.shape[2:]
        k.view(-1, hkv, dim).index_copy_(0, physical.flatten(), key.transpose(1, 2).reshape(-1, hkv, dim))
        v.view(-1, hkv, dim).index_copy_(0, physical.flatten(), value.transpose(1, 2).reshape(-1, hkv, dim))
        self.lengths.copy_((position[-1] + 1).expand(self.batch))
        # Prefill consumes this chunk directly; decode consumes the paged pool.
        return key, value


def _attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
    if attention_mask is not None or dropout:
        raise ValueError("Qwen3.5 adapter supports unpadded inference batches only")
    session = module.paged_decode_session
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    if session.cache.is_prefill:
        output = flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                                 softmax_scale=scaling, causal=True)
    else:
        q = query.squeeze(2)
        k, v = session.cache.pages[module.layer_idx]
        if session.backend == "flash_attn":
            output = flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=session.cache.lengths,
                         block_table=session.cache.tables, softmax_scale=scaling, causal=True)
        elif session.backend == "triton_matched":
            from .matched_splitkv import matched_decode
            output = matched_decode(q, k, v, session.cache.tables, session.cache.lengths,
                                    resources=session.resources, scale=scaling).unsqueeze(1)
        else:
            output = paged_decode(q, k, v, session.cache.tables, session.cache.lengths,
                      resources=session.resources, backend=session.backend, scale=scaling).unsqueeze(1)
        if session.attention_observer is not None:
            session.attention_observer(module, q, k, v, output.squeeze(1))
    return output, None


AttentionInterface.register("paged_decode_lab_qwen35", _attention)


class HybridRunner:
    def __init__(self, model_path, *, backend="flash_attn", mode="eager", batch=1,
                 max_context=2048, device="cuda"):
        validate_hybrid_options(backend, mode, batch, max_context)
        self.backend, self.mode, self.batch = backend, mode, batch
        self.attention_observer = None
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Qwen3.5 hybrid runner requires a CUDA device")
        self.max_context = max_context
        self.model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True, output_loading_info=True)
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError(f"incomplete model loading: {info}")
        self.loading_info = info
        self.model.to(self.device).eval().requires_grad_(False)
        self.text = self.model.model.language_model
        config = self.text.config
        if config.model_type != "qwen3_5_text":
            raise ValueError("Qwen3.5 text backbone required")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.spec = DecodeSpec(batch, config.num_attention_heads, config.num_key_value_heads,
                               config.head_dim, max_context)
        self.cache = PagedHybridCache(config, batch, max_context, self.device)
        self.resources = allocate_hybrid_resources(self.spec, self.device, backend)
        self.linear_implementations = []
        for layer in self.text.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.paged_decode_session = self
                layer.self_attn.config._attn_implementation = "paged_decode_lab_qwen35"
            else:
                self.linear_implementations.append({
                    name: getattr(layer.linear_attn, name).__module__ + "." + getattr(layer.linear_attn, name).__name__
                    for name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule", "causal_conv1d_update")})
        self.graph = None
        if mode == "graph":
            self._capture()

    @torch.inference_mode()
    def _forward(self, tokens, positions):
        # The upstream PyTorch recurrent fallback uses zeros(...).to(value).
        # Make these factories device-local so capture does not perform a
        # synchronous pageable CPU->GPU copy. No model arithmetic is changed.
        with torch.device(self.device):
            return self._forward_impl(tokens, positions)

    def _forward_impl(self, tokens, positions):
        # Unpadded text path: use upstream blocks directly, avoiding host-side mask
        # preparation which reads device cache_position in the HF convenience API.
        hidden = self.text.embed_tokens(tokens)
        position_ids = positions.view(1, 1, -1).expand(3, self.batch, -1)
        embeddings = self.text.rotary_emb(hidden, position_ids)
        for layer in self.text.layers:
            hidden = layer(hidden, position_embeddings=embeddings, attention_mask=None,
                           past_key_values=self.cache, cache_position=positions, use_cache=True)
        return self.model.lm_head(self.text.norm(hidden)[:, -1])

    @torch.inference_mode()
    def _capture(self):
        if self.max_context < 3:
            raise ValueError("Graph warmup needs context capacity >=3")
        self.cache.reset()
        self._forward(torch.ones(self.batch, 2, dtype=torch.long, device=self.device),
                      torch.arange(2, device=self.device))
        self.cache.prefilled = True
        self.cache.is_prefill = False
        self.graph_tokens = torch.ones(self.batch, 1, dtype=torch.long, device=self.device)
        self.graph_position = torch.tensor([2], dtype=torch.long, device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward(self.graph_tokens, self.graph_position)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.graph_logits = self._forward(self.graph_tokens, self.graph_position)
        self.cache.reset()

    @torch.inference_mode()
    def generate(self, prompts, output_tokens=32, *, forced=None, observe=False, timing=False):
        if len(prompts) != self.batch or len({len(p) for p in prompts}) != 1:
            raise ValueError("fixed batch requires equally sized, unpadded token sequences")
        length = len(prompts[0])
        if length < 1 or output_tokens < 1 or length + output_tokens > self.max_context:
            raise ValueError("empty input/output or context capacity exceeded")
        if forced is not None and len(forced) != output_tokens:
            raise ValueError("forced tokens must provide one batch vector per step")
        self.cache.reset()
        inputs = torch.tensor(prompts, device=self.device, dtype=torch.long)
        logits = self._forward(inputs, torch.arange(length, device=self.device))
        self.cache.prefilled = True
        self.cache.is_prefill = False
        history, saved, intervals, predictions = [], [], [], []
        for step in range(output_tokens):
            if step:
                start = end = None
                if timing:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                if self.graph is None:
                    logits = self._forward(token[:, None], torch.tensor([length + step - 1], device=self.device))
                else:
                    self.graph_tokens.copy_(token[:, None])
                    self.graph_position.fill_(length + step - 1)
                    self.graph.replay()
                    logits = self.graph_logits
                if timing:
                    end.record()
                    intervals.append((start, end))
            if observe:
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"nonfinite runner logits at step {step}")
                saved.append(logits.detach().cpu().clone())
            predicted = logits.argmax(-1)
            predictions.append(predicted.clone())
            token = predicted if forced is None else torch.tensor(forced[step], device=self.device, dtype=torch.long)
            history.append(token.clone())
        torch.cuda.synchronize(self.device)
        tokens = torch.stack(history, 1).cpu().tolist()
        predicted_tokens = torch.stack(predictions, 1).cpu().tolist()
        return {"token_ids": tokens, "predicted_ids": predicted_tokens,
                "text": self.tokenizer.batch_decode(tokens, skip_special_tokens=True),
                "logits": saved, "decode_ms": [a.elapsed_time(b) for a, b in intervals]}

    def description(self):
        from dataclasses import asdict
        matched = self.backend == "triton_matched"
        return {"backend": self.backend, "mode": self.mode, "batch": self.batch,
                "decode_contract": "fixed batch and complete block table; no fallback" if matched else "fixed unpadded batch",
                "splitkv_plan": asdict(self.resources.plan) if matched else None,
                "workspace_bytes": (self.resources.partials.numel() + self.resources.lse.numel()) * 4 if matched else
                    self.spec.split_workspace_bytes if self.backend in ("triton_splitk", "triton_tc_splitk") else 0,
                "max_context": self.max_context, "full_attention_layers": len(self.cache.pages),
                "linear_attention_layers": len(self.linear_implementations),
                "heads": self.spec.num_heads, "kv_heads": self.spec.num_kv_heads,
                "head_dim": self.spec.head_dim, "pages_per_full_layer": self.batch * self.cache.width,
                "linear_implementations": self.linear_implementations,
                "graph_scope": "full text decode forward including linear state updates and lm_head; excludes prefill/token selection"}
