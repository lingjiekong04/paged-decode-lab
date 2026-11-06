"""Validated launches and reusable output/split workspace.

Tensor CONTENTS are trusted on the hot path. Call validate_device_metadata once
for externally supplied tables; the nano-vLLM scheduler owns its metadata.
"""
from dataclasses import dataclass
import math
import torch
import triton

from .spec import DecodeSpec, PARTITION_SIZE, TILE_SIZE
from .kernels import _base, _split, _merge


@dataclass
class Resources:
    spec: DecodeSpec
    out: torch.Tensor
    maxima: torch.Tensor | None = None
    denoms: torch.Tensor | None = None
    accums: torch.Tensor | None = None


def allocate_resources(spec, device, backend="triton_splitk"):
    if backend not in ("triton_base", "triton_splitk", "triton_tc", "triton_tc_splitk"):
        raise ValueError("resources require a custom backend")
    out = torch.empty((spec.max_batch, spec.num_heads, spec.head_dim),
                      dtype=torch.bfloat16, device=device)
    result = Resources(spec, out)
    if backend in ("triton_splitk", "triton_tc_splitk"):
        shape = (spec.max_batch, spec.num_heads, spec.partitions)
        result.maxima = torch.empty(shape, device=device, dtype=torch.float32)
        result.denoms = torch.empty(shape, device=device, dtype=torch.float32)
        result.accums = torch.empty((*shape, spec.head_dim), device=device, dtype=torch.float32)
    return result


def validate_device_metadata(table, lengths, num_pages, max_context_len):
    """Debug check: deliberately synchronizes; NEVER call in capture/hot path."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("metadata content validation cannot run during graph capture")
    from .reference import validate_metadata
    validate_metadata(table.cpu().numpy(), lengths.cpu().numpy(), num_pages, max_context_len)


def _validate(q, k, v, tables, lengths, resources, backend, scale):
    if backend not in ("triton_base", "triton_splitk"):
        raise ValueError("unknown custom backend")
    spec = resources.spec
    tensors = (q, k, v, tables, lengths, resources.out)
    if not all(t.is_cuda and t.device == q.device for t in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("custom decode supports BF16 Q/K/V only")
    if q.ndim != 3 or q.shape[1:] != (spec.num_heads, spec.head_dim):
        raise ValueError("Q must have shape [batch, query_heads, head_dim]")
    batch = q.shape[0]
    if not 0 < batch <= spec.max_batch or q.stride(-1) != 1:
        raise ValueError("Q batch exceeds capacity or innermost dimension is not contiguous")
    if k.ndim != 4 or k.shape[1:] != (256, spec.num_kv_heads, spec.head_dim):
        raise ValueError("KV must use NHD pages [pages,256,kv_heads,head_dim]")
    if k.shape != v.shape or not k.is_contiguous() or not v.is_contiguous() or k.shape[0] == 0:
        raise ValueError("K/V must have matching nonempty contiguous storage")
    if tables.ndim != 2 or tables.shape[0] != batch or tables.stride(1) != 1:
        raise ValueError("block table must have shape [batch,capacity] and contiguous rows")
    # During eager execution the scheduler provides only the live table width.
    if not 0 < tables.shape[1] <= spec.table_width:
        raise ValueError("block table width outside configured capacity")
    if lengths.shape != (batch,) or not lengths.is_contiguous():
        raise ValueError("context lengths must be a contiguous [batch] tensor")
    if tables.dtype != torch.int32 or lengths.dtype != torch.int32:
        raise ValueError("block tables and context lengths must be int32")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if (resources.out.shape != (spec.max_batch, spec.num_heads, spec.head_dim)
            or resources.out.dtype != q.dtype or not resources.out.is_contiguous()):
        raise ValueError("invalid output buffer")
    if backend == "triton_splitk":
        shape = (spec.max_batch, spec.num_heads, spec.partitions)
        for tensor, expected in ((resources.maxima, shape), (resources.denoms, shape),
                                 (resources.accums, (*shape, spec.head_dim))):
            if (tensor is None or tensor.shape != expected or tensor.device != q.device
                    or tensor.dtype != torch.float32 or not tensor.is_contiguous()):
                raise ValueError("invalid or missing preallocated split-K workspace")


def paged_decode(q, k, v, tables, lengths, *, resources, backend="triton_base", scale=None):
    """Return borrowed [B,H,D] output. Next launch using resources overwrites it.

    Resources must not overlap live inputs or be used concurrently across streams.
    Live table entries must be valid pages; 0 <= lengths <= max_context_len and
    ceil(length/256) <= table width. Padding table entries may contain -1.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if backend == "triton_tc_splitk":
        from .tc_splitk import tc_splitk_decode
        return tc_splitk_decode(q, k, v, tables, lengths, resources=resources, scale=scale)
    if backend == "triton_tc":
        # Import lazily: precision_decode reuses this module's geometry checks.
        from .experimental_precision import precision_decode
        return precision_decode(q, k, v, tables, lengths, resources=resources,
                                variant="tensorcore_ordered", scale=scale)
    _validate(q, k, v, tables, lengths, resources, backend, scale)
    spec = resources.spec
    b, h, d = q.shape
    common = dict(Q0=q.stride(0), Q1=q.stride(1), TABLE0=tables.stride(0),
                  HQ=h, HKV=spec.num_kv_heads, D=d, SCALE=scale, TILE=TILE_SIZE)
    if backend == "triton_base":
        _base[(b, h)](q, k, v, tables, lengths, resources.out, **common, num_warps=4)
    else:
        _split[(b, h, spec.partitions)](
            q, k, v, tables, lengths, resources.maxima, resources.denoms, resources.accums,
            PARTS=spec.partitions, PART_SIZE=PARTITION_SIZE, **common, num_warps=4)
        _merge[(b, h)](resources.maxima, resources.denoms, resources.accums, resources.out,
                       HQ=h, D=d, PARTS=spec.partitions,
                       PAD_PARTS=triton.next_power_of_2(spec.partitions), num_warps=4)
    return resources.out[:b]
