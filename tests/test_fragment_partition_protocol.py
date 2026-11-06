"""Guard against invalid layouts, missing preflight and uncontrolled kernel edits."""
import ast,copy,importlib.util
from pathlib import Path
from types import SimpleNamespace as T
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    spec=importlib.util.spec_from_file_location('fpart_control',ROOT/'tools/fragment_partition_common.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('d,tile',[(256,64),(64,128),(128,256)])
def test_unsupported_geometry_rejected(module,d,tile):
    with pytest.raises(ValueError,match='D64/TILE256'):
        module.geometry(T(head_dim=d,num_heads=32,num_kv_heads=8),T(tile_tokens=tile),'shared')


def test_unknown_variant_and_missing_prepare(module):
    with pytest.raises(ValueError,match='baseline or shared'):
        module.geometry(None,None,'auto')
    with pytest.raises(ValueError,match='prepare'):
        module.partition(None,None,None,None,None,T(),'shared')


def test_changed_input_identity_rejected(module):
    inputs=tuple(object() for _ in range(5))
    r=T(_fragment_prepared=('shared',inputs,lambda:None,None))
    assert module.partition(*inputs,r,'shared') is None
    with pytest.raises(ValueError,match='input objects changed'):
        module.partition(object(),*inputs[1:],r,'shared')


def test_layout_gate_rejects_blocked_or_wrong_warps(module):
    mma='#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 4]}>'
    shape='tensor<16x256xf32, #mma>'
    call='tt.elementwise_inline_asm "fragment_p[16384]" : ('+shape+', '+shape+', '+shape+') -> '+shape
    module.validate_layout(mma+'\n'+call,256)
    for text in ((mma+'\n'+call).replace('[1, 4]','[4, 1]'), mma+'\n'+call.replace('#mma>','#blocked>')):
        with pytest.raises(ValueError):module.validate_layout(text,256)


def test_kernel_edit_is_limited_to_denominator_and_probe():
    def function(name,fn):
        t=ast.parse((ROOT/'src/paged_decode'/name).read_text(encoding='utf-8-sig'))
        return copy.deepcopy(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==fn))
    a=function('matched_splitkv.py','_normalized_partition')
    b=function('experimental_fragment_partition.py','_fragment_partition')
    for f in (a,b):
        f.name='partition'
        loop=next(n for n in f.body if isinstance(n,ast.For))
        # Only the denominator scaling/update after P and before PV is replaced.
        start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='p' for x in n.targets))
        stop=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='acc' for x in n.targets))
        loop.body[start+1:stop]=[ast.Pass()]
    extras={'Tags','ASM','CONSTRAINTS','PACK','PROBE','TAG_ASM','TAG_CONSTRAINTS'}
    b.args.args=[arg for arg in b.args.args if arg.arg not in extras]
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


@pytest.mark.parametrize('tile',[128,256])
def test_alpha_contraction_matches_selected_compiler_mode(module,tile):
    from fragment_partition_asm import partition_state_asm
    fused,_,pack=partition_state_asm(tile)
    assert fused.count('fma.rn.f32')==2 and 'mul.rn.f32' not in fused
    assert f'fma.rn.f32 a, a, ${3*pack}, x;' in fused
    assert fused.index('fma.rn.f32 a,')<fused.index('add.rn.f32 a,')
    rounded,_,_=partition_state_asm(tile,fp_fusion=False)
    assert 'fma.rn.f32' not in rounded and rounded.count('mul.rn.f32')==2
