"""Independent-reference, page-boundary and Graph tests for the candidate path."""
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
from paged_decode.matched_splitkv import allocate_matched_resources, matched_decode
from paged_decode.spec import DecodeSpec
from paged_decode.torch_reference import dense_reference
from test_gpu import make_gpu_case

pytestmark = pytest.mark.gpu


def close(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(),expected.float(),atol=.02,rtol=.02)


@pytest.mark.parametrize("splits",[3,6])
@pytest.mark.parametrize("dim,hq,hkv",[(64,4,1),(128,4,2),(256,8,2)])
def test_explicit_cross_page_boundaries(splits,dim,hq,hkv):
    lengths=[0,1,255,256,257,1023,1024,2051]
    args=make_gpu_case(lengths,hq,hkv,dim,capacity=2051)
    q,k,v,table,lens=args
    table[0].fill_(-1)
    for b,length in enumerate(lengths[1:],1):
        table[b,(length+255)//256:]=-1
        if length%256:
            page=int(table[b,length//256].item())
            k[page,length%256:]=float("nan")
            v[page,length%256:]=float("nan")
    resources=allocate_matched_resources(DecodeSpec(8,hq,hkv,dim,2051),"cuda",num_splits=splits)
    out=matched_decode(*args,resources=resources)
    close(out,dense_reference(*args))
    assert torch.count_nonzero(out[0])==0


def test_graph_replay_and_empty_partitions():
    args=make_gpu_case([255,1023,0],8,2,256,capacity=2051)
    q,k,v,table,lengths=args
    resources=allocate_matched_resources(DecodeSpec(3,8,2,256,2051),"cuda")
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): matched_decode(*args,resources=resources)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out=matched_decode(*args,resources=resources)
    for values in ([257,1025,0],[2051,1,0],[0,0,0]):
        lengths.copy_(torch.tensor(values,dtype=torch.int32,device="cuda"))
        q.normal_(); k.normal_(); v.normal_(); table.copy_(table.flip(1))
        graph.replay()
        captured=out.clone()
        close(captured,dense_reference(*args))
        eager=matched_decode(*args,resources=resources)
        torch.testing.assert_close(captured,eager,atol=0,rtol=0)


def test_fixed_plan_contract():
    args=make_gpu_case([1,2,0],8,2,256,capacity=2051)
    resources=allocate_matched_resources(DecodeSpec(3,8,2,256,2051),"cuda")
    q,k,v,table,lengths=args
    with pytest.raises(ValueError,match="fixed batch"):
        matched_decode(q[:1],k,v,table[:1],lengths[:1],resources=resources)
    with pytest.raises(ValueError,match="fixed batch"):
        matched_decode(q,k,v,table[:,:8],lengths,resources=resources)
    with pytest.raises(ValueError,match="positive"):
        matched_decode(*args,resources=resources,scale=0)


def test_partition_lse_uses_native_fast_log_protocol():
    """Guard the LSE rounding bug that changed learned-model greedy tokens."""
    from flash_attn import flash_attn_with_kvcache
    args=make_gpu_case([587,587],8,2,256,capacity=1024)
    q,k,v,table,lengths=args
    resources=allocate_matched_resources(DecodeSpec(2,8,2,256,1024),"cuda",num_splits=6)
    matched_decode(*args,resources=resources)
    for part in range(4):
        start=part*192
        end=min(start+192,587)
        pos=torch.arange(start,end,device="cuda")
        pages=table[:,pos//256].long()
        pk,pv=(torch.zeros(2,256,2,256,device="cuda",dtype=torch.bfloat16) for _ in range(2))
        pk[:,:end-start]=k[pages,pos[None,:]%256]
        pv[:,:end-start]=v[pages,pos[None,:]%256]
        _,lse=flash_attn_with_kvcache(q[:,None],pk,pv,
            block_table=torch.arange(2,device="cuda",dtype=torch.int32)[:,None],
            cache_seqlens=torch.full_like(lengths,end-start),num_splits=1,
            causal=True,return_softmax_lse=True)
        torch.testing.assert_close(resources.lse[:,:,part],lse.reshape(2,8),atol=0,rtol=0)
