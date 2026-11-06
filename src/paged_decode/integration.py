"""Explicit nano-vLLM initialization hook; no monkeypatching."""
import torch

from .runtime import allocate_resources
from .spec import DecodeSpec


def configure_model(model, config):
    if config.decode_backend == "flash_attn":
        return
    # Layers execute sequentially on one stream; matching geometries share scratch.
    # Projection consumes each borrowed attention output before the next layer.
    shared = {}
    for module in model.modules():
        if not all(hasattr(module, x) for x in ("k_cache", "v_cache", "num_heads", "num_kv_heads", "head_dim")):
            continue
        if module.k_cache.dtype != torch.bfloat16:
            raise ValueError("custom nano-vLLM decode requires a BF16 model")
        spec = DecodeSpec(config.max_num_seqs, module.num_heads, module.num_kv_heads,
                          module.head_dim, config.max_model_len, config.kvcache_block_size)
        if spec not in shared:
            shared[spec] = allocate_resources(spec, module.k_cache.device, config.decode_backend)
        module.decode_backend = config.decode_backend
        module.decode_resources = shared[spec]
    if not shared:
        raise ValueError("no compatible attention layers found")
