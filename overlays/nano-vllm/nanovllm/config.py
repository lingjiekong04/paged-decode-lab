import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    decode_backend: str = "flash_attn"
    # Exact KV pool for controlled experiments; None retains upstream sizing.
    kv_cache_blocks: int | None = None
    record_decode_timings: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        from paged_decode.spec import validate_engine_options
        validate_engine_options(self.decode_backend, self.tensor_parallel_size,
                                self.kvcache_block_size, self.max_num_seqs, self.max_model_len)
        if self.kv_cache_blocks is not None and self.kv_cache_blocks <= 0:
            raise ValueError("kv_cache_blocks must be positive")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.decode_backend != "flash_attn":
            from paged_decode.spec import DecodeSpec
            if self.hf_config.model_type != "qwen3":
                raise ValueError("custom engine integration supports dense Qwen3 only")
            if self.hf_config.dtype not in ("bfloat16", __import__("torch").bfloat16):
                raise ValueError("custom engine integration requires BF16 weights")
            DecodeSpec(self.max_num_seqs, self.hf_config.num_attention_heads,
                       self.hf_config.num_key_value_heads, self.hf_config.head_dim,
                       self.max_model_len, self.kvcache_block_size)
