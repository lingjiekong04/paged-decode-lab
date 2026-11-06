import numpy as np
import pytest

from paged_decode.reference import dense_reference, streaming_reference, validate_metadata, merge_partials
from paged_decode.spec import DecodeSpec, validate_engine_options


def make_case(lengths, hq=4, hkv=1, dim=64, seed=9, capacity=None):
    rng = np.random.default_rng(seed)
    capacity = max(max(lengths), 1) if capacity is None else capacity
    pages_per_row = (capacity + 255) // 256
    pages = len(lengths) * pages_per_row
    table = rng.permutation(pages).reshape(len(lengths), pages_per_row).astype(np.int32)
    k = rng.normal(size=(pages, 256, hkv, dim)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    q = rng.normal(size=(len(lengths), hq, dim)).astype(np.float32)
    return q, k, v, table, np.array(lengths, dtype=np.int32)


@pytest.mark.parametrize("length", [0, 1, 255, 256, 257, 1023, 1024, 1025, 2051])
@pytest.mark.parametrize("dim,hq,hkv", [(64, 4, 1), (128, 4, 2), (64, 2, 2), (256, 8, 2)])
def test_streaming_and_split_match_independent_dense(length, dim, hq, hkv):
    args = make_case([length, 1], hq, hkv, dim)
    expected = dense_reference(*args)
    for split in (False, True):
        actual = streaming_reference(*args, split=split)
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
        assert np.isfinite(actual).all()


def test_partial_page_poison_and_padding_are_ignored():
    args = list(make_case([257, 0], capacity=1025))
    table = args[3]
    # Poison all unused pages/slots; live attention must not observe them.
    args[1][table[0, 1], 1:] = np.nan
    args[2][table[0, 1], 1:] = np.nan
    table[0, 2:] = -1
    table[1] = -1
    expected = dense_reference(*args)
    for split in (False, True):
        actual = streaming_reference(*args, split=split)
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
        assert not actual[1].any()


def test_large_logits_stable_softmax():
    args = list(make_case([1025], seed=23))
    args[0] *= 80
    args[1] *= 80
    expected = dense_reference(*args)
    np.testing.assert_allclose(streaming_reference(*args, split=True), expected, atol=1e-4)


@pytest.mark.parametrize("length,table", [(-1, [[0]]), (257, [[0]]), (1, [[-1]]), (1, [[2]])])
def test_invalid_metadata_rejected(length, table):
    with pytest.raises(ValueError):
        validate_metadata(np.array(table), np.array([length]), 2, 256)


def test_spec_and_memory():
    spec = DecodeSpec(32, 32, 8, 128, 8192)
    assert spec.partitions == 8
    assert spec.table_width == 32
    assert spec.split_workspace_bytes == 32 * 32 * 8 * 130 * 4
    np.testing.assert_array_equal(merge_partials([], 64), np.zeros(64))


@pytest.mark.parametrize("kwargs", [{"head_dim": 80}, {"page_size": 16}, {"num_kv_heads": 3},
                                    {"max_batch": 0}, {"max_context_len": -1}])
def test_unsupported_geometry(kwargs):
    values = dict(max_batch=8, num_heads=4, num_kv_heads=2, head_dim=64, max_context_len=2048)
    values.update(kwargs)
    with pytest.raises(ValueError):
        DecodeSpec(**values)


@pytest.mark.parametrize("backend,tp,page", [("typo", 1, 256), ("triton_base", 2, 256),
                                            ("triton_splitk", 1, 512)])
def test_engine_rejects_unsupported_options(backend, tp, page):
    with pytest.raises(ValueError):
        validate_engine_options(backend, tp, page, 32, 8192)
