"""Qwen3.5 cache mechanics; requires the separate Transformers 5.3 environment."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5 or not torch.cuda.is_available():
    pytest.skip("Qwen3.5 cache tests require Transformers 5 and CUDA", allow_module_level=True)

from paged_decode.hybrid import PagedHybridCache, StableStates

pytestmark = pytest.mark.gpu


def test_hybrid_prefill_and_decode_write_random_pages():
    config = SimpleNamespace(num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
                             num_key_value_heads=2, head_dim=256)
    cache = PagedHybridCache(config, 2, 1025, "cuda")
    key = torch.randn(2, 2, 257, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    position = torch.arange(257, device="cuda")
    cache.update(key, value, 3, {"cache_position": position})
    k, v = cache.pages[3]
    for row in range(2):
        physical = cache.tables[row, position // 256].long()
        torch.testing.assert_close(k[physical, position % 256], key[row].transpose(0, 1), atol=0, rtol=0)
        torch.testing.assert_close(v[physical, position % 256], value[row].transpose(0, 1), atol=0, rtol=0)
    new_key = torch.randn(2, 2, 1, 256, device="cuda", dtype=torch.bfloat16)
    cache.update(new_key, new_key, 3, {"cache_position": torch.tensor([257], device="cuda")})
    assert cache.lengths.tolist() == [258, 258]
    for row in range(2):
        torch.testing.assert_close(k[cache.tables[row, 1].long(), 1], new_key[row, :, 0], atol=0, rtol=0)
    cache.prefilled = True
    cache.reset()
    assert not cache.has_previous_state
    assert cache.lengths.tolist() == [0, 0]


def test_recurrent_state_pointer_survives_graph_replay():
    states = StableStates([None])
    states[0] = torch.zeros(2, 3, device="cuda")
    pointer = states[0].data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        states[0] = states[0] + 1
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        states[0] = states[0] + 1
    states[0].zero_()
    for _ in range(4):
        graph.replay()
    assert states[0].data_ptr() == pointer
    torch.testing.assert_close(states[0], torch.full_like(states[0], 4))
    with pytest.raises(ValueError):
        states[0] = torch.zeros(3, 3, device="cuda")
