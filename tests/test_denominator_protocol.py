"""The selection experiment must preserve launch geometry and four-warps policy."""
import importlib.util,sys,types
from pathlib import Path
import pytest


def load(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'tools'))
    spec=importlib.util.spec_from_file_location('denominator_control_test',root/'tools/denominator_ablation_common.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_only_kernel_implementation_changes(monkeypatch):
    records=[];merged=[]
    class Kernel:
        def __init__(self,name):self.name=name
        def __getitem__(self,grid):
            def call(*args,**kwargs):records.append((self.name,grid,args,kwargs))
            return call
    a=types.ModuleType('paged_decode.matched_splitkv');a._normalized_partition=Kernel('baseline')
    b=types.ModuleType('paged_decode.experimental_denominator');b._partition_select=Kernel('select')
    monkeypatch.setitem(sys.modules,'paged_decode.matched_splitkv',a)
    monkeypatch.setitem(sys.modules,'paged_decode.experimental_denominator',b)
    m=load(monkeypatch);monkeypatch.setattr(m,'merge',lambda r:merged.append(r))
    T=types.SimpleNamespace
    r=T(spec=T(max_batch=1,num_heads=32,num_kv_heads=8,head_dim=64),
        plan=T(splits=16,tokens_per_split=512,tile_tokens=256),partials=object(),lse=object(),out=object())
    q=T(stride=lambda i:2048 if i==0 else 64);table=T(stride=lambda i:32)
    inputs=(q,object(),object(),table,object())
    for v in ('baseline','select'):assert m.complete(*inputs,r,v) is r.out
    assert records[0][0]=='baseline' and records[1][0]=='select'
    assert records[0][1:]==records[1][1:]
    assert records[0][1]==(1,8,16) and records[0][3]['num_warps']==4
    assert merged==[r,r]


@pytest.mark.parametrize('variant',['auto',4,None])
def test_unknown_candidate_is_rejected(monkeypatch,variant):
    with pytest.raises(ValueError,match='baseline or select'):load(monkeypatch).partition(None,None,None,None,None,None,variant)
