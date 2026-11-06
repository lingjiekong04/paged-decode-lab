"""Check that the control experiment changes only the partition launch warps."""
import importlib.util,sys,types
from pathlib import Path
import pytest


def load():
    path=Path(__file__).resolve().parents[1]/'tools/warp_ablation_common.py'
    spec=importlib.util.spec_from_file_location('warp_ablation_common_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_control_preserves_partition_plan_and_merge(monkeypatch):
    records=[]
    class Kernel:
        def __init__(self,name):self.name=name
        def __getitem__(self,grid):
            def call(*args,**kwargs):records.append((self.name,grid,args,kwargs));return self.name
            return call
    fake=types.ModuleType('paged_decode.matched_splitkv')
    fake._normalized_partition=Kernel('partition');fake._lse_merge=Kernel('merge')
    monkeypatch.setitem(sys.modules,'paged_decode.matched_splitkv',fake)
    T=types.SimpleNamespace
    spec=T(max_batch=2,num_heads=32,num_kv_heads=8,head_dim=128)
    plan=T(splits=6,tokens_per_split=384,tile_tokens=128,lse_lanes=4)
    resources=T(spec=spec,plan=plan,partials=object(),lse=object(),out=object())
    q=T(stride=lambda dim:4096 if dim==0 else 128);table=T(stride=lambda dim:9)
    inputs=(q,object(),object(),table,object())
    adapter=load()
    for warps in (4,8):assert adapter.complete(*inputs,resources,warps) is resources.out
    p4,m4,p8,m8=records
    assert p4[1]==p8[1]==(2,8,6)
    assert p4[2]==p8[2] and p4[3]['num_warps']==4 and p8[3]['num_warps']==8
    assert {k:v for k,v in p4[3].items() if k!='num_warps'}=={k:v for k,v in p8[3].items() if k!='num_warps'}
    assert p4[3]['PART_SIZE']==384 and p4[3]['TILE']==128
    assert m4==m8 and m4[1]==(2,32) and m4[3]['num_warps']==4


@pytest.mark.parametrize('warps',[0,2,16,True,4.0])
def test_invalid_launch_configuration_is_rejected_before_gpu_import(warps):
    with pytest.raises(ValueError,match='4 or 8'):load().partition(None,None,None,None,None,None,warps)
