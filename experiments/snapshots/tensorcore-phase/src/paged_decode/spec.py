"""Host-side geometry contract; no CUDA imports or device synchronization."""
from dataclasses import dataclass
import math

BACKENDS = ("flash_attn", "triton_base", "triton_splitk", "triton_tc")
PAGE_SIZE = 256
PARTITION_SIZE = 1024
TILE_SIZE = 64


@dataclass(frozen=True)
class DecodeSpec:
    max_batch: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_context_len: int
    page_size: int = PAGE_SIZE

    def __post_init__(self):
        for field in ("max_batch", "num_heads", "num_kv_heads", "max_context_len"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.head_dim not in (64, 128, 256):
            raise ValueError("custom decode supports head_dim 64, 128, or 256")
        if self.page_size != PAGE_SIZE:
            raise ValueError("custom decode supports page_size=256 only")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")

    @property
    def partitions(self):
        return math.ceil(self.max_context_len / PARTITION_SIZE)

    @property
    def table_width(self):
        return math.ceil(self.max_context_len / self.page_size)

    @property
    def split_workspace_bytes(self):
        return self.max_batch * self.num_heads * self.partitions * (self.head_dim + 2) * 4


def validate_engine_options(backend, tp, page_size, max_batch, max_context_len):
    if backend not in BACKENDS:
        raise ValueError(f"unknown decode_backend {backend!r}; choose {BACKENDS}")
    if max_batch <= 0 or max_context_len <= 0:
        raise ValueError("batch and context limits must be positive")
    if backend != "flash_attn":
        if tp != 1:
            raise ValueError("custom decode supports tensor_parallel_size=1 only")
        if page_size != PAGE_SIZE:
            raise ValueError("custom decode requires kvcache_block_size=256")
