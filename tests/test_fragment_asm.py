"""CPU geometry/assembly safety checks; actual slot ownership needs the GPU probe."""
import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def asm():
    p=Path(__file__).resolve().parents[1]/'tools/fragment_asm.py'
    s=importlib.util.spec_from_file_location('fragment_asm_test',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


@pytest.mark.parametrize('tile',[128,256])
def test_coordinates_cover_each_element_once(asm,tile):
    values=[asm.coordinates(t,s,tile) for t in range(128) for s in range(tile//8)]
    assert len(set(values))==len(values)==16*tile
    assert set(values)=={(r,c) for r in range(16) for c in range(tile)}
    for thread in range(128):
        positions=[asm.coordinates(thread,s,tile) for s in range(tile//8)]
        assert {c//2%4 for _,c in positions}=={thread%4}


@pytest.mark.parametrize('tile',[128,256])
def test_shared_barriers_and_order(asm,tile):
    code,constraints,pack=asm.shared_ordered_asm(tile)
    assert pack==tile//8
    assert code.count('bar.sync 0;')==2
    assert code.index('bar.sync 0;')<code.index('@!leader bra FRAGMENT_JOIN;')
    assert code.index('FRAGMENT_JOIN:')<code.rindex('bar.sync 0;')
    assert code.count('add.rn.f32 a, a, x;')==tile//4
    assert code.count('add.rn.f32 b, b, y;')==tile//4
    assert constraints.endswith('~{memory}')
    # Read positions must be block0-even, block0-odd, block1-even, ... .
    import re
    offsets=list(map(int,re.findall(r'ld.shared.f32 x, \[addr\+(\d+)\];',code)))
    assert offsets==[(8*b+parity)*4 for b in range(tile//8) for parity in range(2)]
    assert code.count('shfl.sync.idx.b32 u, pair, src0')==2
    assert code.count('shfl.sync.idx.b32 v, pair, src1')==2


@pytest.mark.parametrize('tile',[0,64,512])
def test_unsupported_geometry_is_rejected(asm,tile):
    with pytest.raises(ValueError):asm.coordinates(0,0,tile)
    with pytest.raises(ValueError):asm.shared_ordered_asm(tile)
    with pytest.raises(ValueError):asm.marker_asm(tile)


@pytest.mark.parametrize('tile',[128,256])
def test_state_output_precedes_final_merge(asm,tile):
    code,_,_=asm.shared_ordered_asm(tile,states=True)
    assert '.b8 fragment_sum[256]' in code
    assert 'shfl.sync.' not in code
    assert code.count('bar.sync 0;')==2
    assert 'st.shared.f32 [outaddr+128], b;' in code
