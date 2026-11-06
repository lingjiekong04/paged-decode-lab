import importlib.util
from pathlib import Path
from types import SimpleNamespace as T
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    import fragment_asm,fragment_asm_d256,fragment_d256_common,fragment_partition_asm_d256
    return fragment_asm,fragment_asm_d256,fragment_d256_common,fragment_partition_asm_d256


def test_d256_fragment_bijection(modules):
    _,asm,_,_=modules
    mapping=[asm.coordinates(t,s,64) for t in range(128) for s in range(8)]
    assert len(set(mapping))==1024
    assert set(mapping)=={(r,c) for r in range(16) for c in range(64)}


@pytest.mark.parametrize('tile',[128,256])
def test_previous_geometries_byte_identical(modules,tile):
    old,new,_,_=modules
    assert old.marker_asm(tile)==new.marker_asm(tile)
    assert old.shared_ordered_asm(tile,states=True)==new.shared_ordered_asm(tile,states=True)


def test_d256_geometry_and_contraction(modules):
    _,_,common,asm=modules
    common.geometry(T(head_dim=256,num_heads=16,num_kv_heads=4),T(tile_tokens=64),'shared')
    text,_,pack=asm.partition_state_asm(64)
    assert pack==8 and text.count('fma.rn.f32')==2
    assert 'fma.rn.f32 a, a, $24, x;' in text
    assert 'fma.rn.f32 b, b, $26, y;' in text


@pytest.mark.parametrize('tile',[32,128,256])
def test_d256_wrong_tile_rejected(modules,tile):
    _,_,common,_=modules
    with pytest.raises(ValueError):common.geometry(T(head_dim=256,num_heads=16,num_kv_heads=4),T(tile_tokens=tile),'shared')
