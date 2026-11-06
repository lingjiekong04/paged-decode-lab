"""Static split-KV plan for the pinned FlashAttention 2.7.4 inference geometry.

Single query, BF16, D=64/128/256, no window/ALiBi. This is a deterministic
geometry policy, not runtime benchmarking/autotuning. See THIRD_PARTY.md.
"""
from dataclasses import dataclass
import math
import struct


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


@dataclass(frozen=True)
class SplitKVPlan:
    splits: int
    tile_tokens: int
    tokens_per_split: int
    table_capacity_tokens: int
    lse_lanes: int
    sm_count: int
    policy: str


def make_splitkv_plan(spec, sm_count, *, num_splits=0):
    if type(sm_count) is not int or sm_count < 1:
        raise ValueError("positive SM count required")
    if type(num_splits) is not int or not 0 <= num_splits <= 128:
        raise ValueError("num_splits must be an integer in [0,128]")
    if spec.num_heads // spec.num_kv_heads > 16:
        raise ValueError("matched split-KV requires at most 16 query heads per KV head")
    tile = 256 if spec.head_dim == 64 else (128 if spec.head_dim == 128 else 64)
    capacity = spec.table_width * 256
    blocks = (capacity + tile - 1) // tile
    splits = num_splits
    if not splits:
        work = spec.max_batch * spec.num_kv_heads
        virtual_sms = sm_count * 2
        if work >= _f32(_f32(.8) * virtual_sms):
            splits = 1
        else:
            maximum = min(128, virtual_sms, blocks)
            eligible = lambda n: n == 1 or math.ceil(blocks / n) != math.ceil(blocks / (n - 1))
            efficiency = {}
            for n in range(1, maximum + 1):
                if eligible(n):
                    waves = _f32(_f32(work * n) / virtual_sms)
                    efficiency[n] = _f32(waves / math.ceil(waves))
            best = max(efficiency.values())
            splits = next(n for n, value in efficiency.items() if value >= .85 * best)
    padded = 1 << (splits - 1).bit_length()
    merge_rows = 4 if spec.head_dim % 128 == 0 else 8
    return SplitKVPlan(splits, tile, math.ceil(blocks / splits) * tile, capacity,
                       min(padded, 128 // merge_rows), sm_count,
                       "explicit" if num_splits else "pinned_geometry")
