"""Independent-oracle and replay tests for experimental BF16-exp kernels."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA device unavailable", allow_module_level=True)

from paged_decode.experimental_precision import VARIANTS, precision_decode
from paged_decode.runtime import allocate_resources
from paged_decode.spec import DecodeSpec
from paged_decode.torch_reference import dense_reference
from test_gpu import make_gpu_case

pytestmark = pytest.mark.gpu


def check(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dim,hq,hkv", [(64, 4, 1), (128, 4, 2), (256, 8, 2), (64, 2, 2)])
def test_precision_boundaries_and_nan_padding(variant, dim, hq, hkv):
    lengths = [0, 1, 255, 256, 257, 1023, 1024, 1025]
    args = make_gpu_case(lengths, hq, hkv, dim, capacity=1025)
    q, k, v, table, lens = args
    table[0].fill_(-1)
    for b, length in enumerate(lengths[1:], 1):
        table[b, (length + 255) // 256:] = -1
        if length % 256:
            page = int(table[b, length // 256].item())
            k[page, length % 256:] = float("nan")
            v[page, length % 256:] = float("nan")
    resources = allocate_resources(DecodeSpec(8, hq, hkv, dim, 1025), "cuda", "triton_base")
    actual = precision_decode(*args, resources=resources, variant=variant)
    check(actual, dense_reference(*args))
    assert torch.count_nonzero(actual[0]) == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_precision_graph_metadata_and_kv_updates(variant):
    args = make_gpu_case([255, 1023, 0], 8, 2, 256, capacity=2048)
    q, k, v, table, lens = args
    resources = allocate_resources(DecodeSpec(3, 8, 2, 256, 2048), "cuda", "triton_base")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            precision_decode(*args, resources=resources, variant=variant)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = precision_decode(*args, resources=resources, variant=variant)
    for lengths in ([256, 1024, 0], [257, 1025, 1], [1, 0, 1025], [0, 0, 0]):
        lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        q.normal_()
        k.normal_()
        v.normal_()
        table.copy_(table.flip(1))
        graph.replay()
        check(output, dense_reference(*args))


def test_precision_unsupported_variants_and_groups_fail_explicitly():
    args = make_gpu_case([1], 32, 1, 64, capacity=256)
    resources = allocate_resources(DecodeSpec(1, 32, 1, 64, 256), "cuda", "triton_base")
    with pytest.raises(ValueError, match="variant"):
        precision_decode(*args, resources=resources, variant="unknown")
    with pytest.raises(ValueError, match="16"):
        precision_decode(*args, resources=resources, variant="tensorcore_bf16")
