"""Independent dense oracle and executable NumPy model of streaming/split softmax.

This is an algorithm check, never evidence that a Triton kernel executed.
"""
import numpy as np

from .spec import PARTITION_SIZE, TILE_SIZE


def validate_metadata(table, lengths, num_pages, max_context_len, page_size=256):
    table, lengths = np.asarray(table), np.asarray(lengths)
    if table.ndim != 2 or lengths.ndim != 1 or len(table) != len(lengths):
        raise ValueError("metadata batch dimensions differ")
    if not np.issubdtype(table.dtype, np.integer) or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("metadata must be integral")
    if np.any(lengths < 0) or np.any(lengths > max_context_len):
        raise ValueError("context length outside configured capacity")
    for row, length in zip(table, lengths):
        count = (int(length) + page_size - 1) // page_size
        if count > len(row) or np.any(row[:count] < 0) or np.any(row[:count] >= num_pages):
            raise ValueError("invalid live page mapping")


def gather_kv(cache, table_row, length):
    if length == 0:
        return np.empty((0, *cache.shape[2:]), dtype=np.float32)
    positions = np.arange(length)
    return cache[table_row[positions // cache.shape[1]], positions % cache.shape[1]].astype(np.float32)


def dense_reference(q, k, v, table, lengths, scale=None):
    q = np.asarray(q, dtype=np.float32)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    validate_metadata(table, lengths, len(k), table.shape[1] * k.shape[1], k.shape[1])
    result = np.zeros_like(q)
    group = q.shape[1] // k.shape[2]
    for b, length in enumerate(lengths):
        if not length:
            continue
        keys = gather_kv(k, table[b], int(length))
        vals = gather_kv(v, table[b], int(length))
        for h in range(q.shape[1]):
            scores = (keys[:, h // group] @ q[b, h]) * scale
            weights = np.exp(scores - scores.max())
            result[b, h] = weights @ vals[:, h // group] / weights.sum()
    return result


def partial(q, keys, vals, scale, tile=TILE_SIZE):
    m, l = np.float32(-np.inf), np.float32(0)
    acc = np.zeros(q.shape, dtype=np.float32)
    for start in range(0, len(keys), tile):
        scores = (keys[start:start + tile] @ q) * scale
        next_m = np.maximum(m, scores.max())
        alpha = np.exp(m - next_m)
        p = np.exp(scores - next_m)
        acc = acc * alpha + p @ vals[start:start + tile]
        l = l * alpha + p.sum()
        m = next_m
    return m, l, acc


def merge_partials(parts, dim):
    live = [(m, l, a) for m, l, a in parts if l > 0]
    if not live:
        return np.zeros(dim, dtype=np.float32)
    m = max(p[0] for p in live)
    factors = [np.exp(p[0] - m) for p in live]
    numerator = sum(f * p[2] for f, p in zip(factors, live))
    denominator = sum(f * p[1] for f, p in zip(factors, live))
    return numerator / denominator


def streaming_reference(q, k, v, table, lengths, split=False, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    validate_metadata(table, lengths, len(k), table.shape[1] * k.shape[1], k.shape[1])
    result = np.zeros(q.shape, dtype=np.float32)
    group = q.shape[1] // k.shape[2]
    for b, length in enumerate(lengths):
        keys = gather_kv(k, table[b], int(length))
        vals = gather_kv(v, table[b], int(length))
        for h in range(q.shape[1]):
            step = PARTITION_SIZE if split else max(1, int(length))
            parts = [partial(q[b, h].astype(np.float32), keys[s:s + step, h // group],
                             vals[s:s + step, h // group], scale)
                     for s in range(0, int(length), step)]
            result[b, h] = merge_partials(parts, q.shape[-1])
    return result
