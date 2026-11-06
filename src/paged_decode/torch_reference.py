"""FP32 gather + dense softmax oracle; intentionally separate from kernel math."""
import torch


def dense_reference(q, k, v, tables, lengths):
    result = torch.zeros_like(q, dtype=torch.float32)
    group = q.shape[1] // k.shape[2]
    for b, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            continue
        positions = torch.arange(length, device=q.device)
        pages = tables[b, positions // 256].long()
        keys = k[pages, positions % 256].float().repeat_interleave(group, dim=1)
        vals = v[pages, positions % 256].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[b].float(), keys) * q.shape[-1] ** -0.5
        result[b] = torch.einsum("hn,nhd->hd", scores.softmax(-1), vals)
    return result
