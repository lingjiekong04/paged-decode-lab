"""Verify all selected blocks reconstruct token order for every supported tile."""
import pytest
torch=pytest.importorskip('torch')
triton=pytest.importorskip('triton')
import triton.language as tl
if not torch.cuda.is_available():pytest.skip('CUDA required',allow_module_level=True)
from paged_decode.experimental_denominator import _select_block
pytestmark=pytest.mark.gpu


@triton.jit
def _selection_probe(X,Y,TILE:tl.constexpr):
    r=tl.arange(0,16);t=tl.arange(0,TILE);d=tl.arange(0,8)
    values=tl.load(X+r[:,None]*TILE+t[None,:])
    for block in tl.static_range(TILE//8):
        selected=_select_block(tl.reshape(values,(16,TILE//8,8)),TILE//8,block)
        tl.store(Y+r[:,None]*TILE+block*8+d[None,:],selected)


@pytest.mark.parametrize('tile',[64,128,256])
def test_each_block_preserves_order(tile):
    x=torch.arange(16*tile,device='cuda',dtype=torch.float32).reshape(16,tile)
    y=torch.full_like(x,float('nan'))
    _selection_probe[(1,)](x,y,TILE=tile,num_warps=4)
    torch.testing.assert_close(x,y,atol=0,rtol=0)
