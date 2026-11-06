"""Real CUDA tests. A skipped module is NOT a passed GPU validation."""
import os
import pytest

torch = pytest.importorskip("torch", reason="GPU suite requires torch")
pytest.importorskip("triton", reason="GPU suite requires Triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA device unavailable", allow_module_level=True)

from paged_decode.runtime import allocate_resources, paged_decode, validate_device_metadata
from paged_decode.spec import DecodeSpec

pytestmark = pytest.mark.gpu


def make_gpu_case(lengths, hq=4, hkv=1, dim=64, capacity=2048):
    torch.manual_seed(19)
    batch = len(lengths)
    width = (capacity + 255) // 256
    pages = batch * width
    table = torch.randperm(pages, device="cuda").to(torch.int32).reshape(batch, width)
    k = torch.randn(pages, 256, hkv, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(batch, hq, dim, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return q, k, v, table, lens


def torch_reference(q, k, v, tables, lengths):
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


def assert_close_report(actual, expected):
    error = (actual.float() - expected.float()).abs()
    print({"max_abs_error": error.max().item(), "mean_squared_error": error.square().mean().item()})
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("backend", ["triton_base", "triton_splitk", "triton_tc"])
@pytest.mark.parametrize("dim,hq,hkv", [(64, 4, 1), (128, 4, 2), (64, 2, 2), (256, 8, 2)])
def test_boundaries_gqa_and_padding(backend, dim, hq, hkv):
    lengths = [0, 1, 255, 256, 257, 1023, 1024, 1025]
    args = make_gpu_case(lengths, hq, hkv, dim, capacity=1025)
    q, k, v, table, lens = args
    table[0].fill_(-1)
    for b, length in enumerate(lengths[1:], 1):
        table[b, (length + 255) // 256:] = -1
        if length % 256:
            page = table[b, length // 256].item()
            k[page, length % 256:] = float("nan")
            v[page, length % 256:] = float("nan")
    validate_device_metadata(table, lens, len(k), 1025)
    resources = allocate_resources(DecodeSpec(len(lengths), hq, hkv, dim, 1025), "cuda", backend)
    actual = paged_decode(*args, resources=resources, backend=backend).clone()
    assert_close_report(actual, torch_reference(*args))
    assert torch.count_nonzero(actual[0]) == 0


@pytest.mark.parametrize("backend", ["triton_base", "triton_splitk", "triton_tc"])
@pytest.mark.parametrize("dim", [64, 256])
def test_graph_replay_changes_lengths_pages_and_kv(backend, dim):
    args = make_gpu_case([255, 1023, 0], dim=dim, capacity=2048)
    q, k, v, table, lens = args
    resources = allocate_resources(DecodeSpec(3, 4, 1, dim, 2048), "cuda", backend)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            paged_decode(*args, resources=resources, backend=backend)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = paged_decode(*args, resources=resources, backend=backend)
    for lengths in ([256, 1024, 0], [257, 1025, 1], [1, 0, 1025], [0, 0, 0]):
        lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        q.normal_()
        k.normal_()
        v.normal_()
        table.copy_(table.flip(1))
        graph.replay()
        assert_close_report(output, torch_reference(*args))
        eager = paged_decode(*args, resources=resources, backend=backend).clone()
        assert_close_report(eager, torch_reference(*args))


@pytest.mark.parametrize("backend", ["triton_base", "triton_splitk", "triton_tc"])
@pytest.mark.parametrize("dim", [128, 256])
def test_flash_attention_peer(backend, dim):
    flash = pytest.importorskip("flash_attn")
    args = make_gpu_case([1, 257, 1025], dim=dim)
    q, k, v, table, lens = args
    resources = allocate_resources(DecodeSpec(3, 4, 1, dim, 2048), "cuda", backend)
    actual = paged_decode(*args, resources=resources, backend=backend).clone()
    peer = flash.flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=lens,
                                        block_table=table, causal=True).squeeze(1)
    assert_close_report(actual, peer)


def test_public_contract_rejects_bad_dtype_and_missing_workspace():
    args = make_gpu_case([2])
    r = allocate_resources(DecodeSpec(1, 4, 1, 64, 2048), "cuda", "triton_base")
    with pytest.raises(ValueError, match="workspace"):
        paged_decode(*args, resources=r, backend="triton_splitk")
    with pytest.raises(ValueError, match="BF16"):
        paged_decode(args[0].float(), *args[1:], resources=r)


@pytest.mark.parametrize("dim", [64, 256])
def test_non_power_of_two_split_count(dim):
    # Three real partitions reduce in a four-lane padded merge tile.
    args = make_gpu_case([0, 2051, 2049, 1025], dim=dim, capacity=2051)
    resources = allocate_resources(DecodeSpec(4, 4, 1, dim, 2051), "cuda", "triton_splitk")
    actual = paged_decode(*args, resources=resources, backend="triton_splitk")
    assert_close_report(actual, torch_reference(*args))


def test_sampler_zero_temperature_is_deterministic():
    from nanovllm.layers.sampler import Sampler
    logits = torch.tensor([[1., 3., 2.], [4., 1., 2.]], device="cuda")
    sampler = Sampler()
    for _ in range(3):
        result = sampler(logits.clone(), torch.zeros(2, device="cuda"))
        torch.testing.assert_close(result, logits.argmax(-1))
