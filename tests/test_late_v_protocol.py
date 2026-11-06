"""The selection experiment must preserve launch geometry and four-warps policy."""
import importlib.util,sys,types
from pathlib import Path
import pytest


def load(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'tools'))
    spec=importlib.util.spec_from_file_location('late_v_control_test',root/'tools/late_v_ablation_common.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_only_kernel_implementation_changes(monkeypatch):
    records=[];merged=[]
    class Kernel:
        def __init__(self,name):self.name=name
        def __getitem__(self,grid):
            def call(*args,**kwargs):records.append((self.name,grid,args,kwargs))
            return call
    a=types.ModuleType('paged_decode.matched_splitkv');a._normalized_partition=Kernel('baseline')
    b=types.ModuleType('paged_decode.experimental_late_v');b._partition_late_v=Kernel('late_v')
    monkeypatch.setitem(sys.modules,'paged_decode.matched_splitkv',a)
    monkeypatch.setitem(sys.modules,'paged_decode.experimental_late_v',b)
    m=load(monkeypatch);monkeypatch.setattr(m,'merge',lambda r:merged.append(r))
    T=types.SimpleNamespace
    r=T(spec=T(max_batch=1,num_heads=32,num_kv_heads=8,head_dim=64),
        plan=T(splits=16,tokens_per_split=512,tile_tokens=256),partials=object(),lse=object(),out=object())
    q=T(stride=lambda i:2048 if i==0 else 64);table=T(stride=lambda i:32)
    inputs=(q,object(),object(),table,object())
    for v in ('baseline','late_v'):assert m.complete(*inputs,r,v) is r.out
    assert records[0][0]=='baseline' and records[1][0]=='late_v'
    assert records[0][1:]==records[1][1:]
    assert records[0][1]==(1,8,16) and records[0][3]['num_warps']==4
    assert merged==[r,r]


@pytest.mark.parametrize('variant',['auto',4,None])
def test_unknown_candidate_is_rejected(monkeypatch,variant):
    with pytest.raises(ValueError,match='baseline or late_v'):load(monkeypatch).partition(None,None,None,None,None,None,variant)


def test_only_v_load_position_changes():
    import ast,copy
    root=Path(__file__).resolve().parents[1]/'src/paged_decode'
    def get(name,function):
        tree=ast.parse((root/name).read_text(encoding='utf-8-sig'))
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    original=get('matched_splitkv.py','_normalized_partition');candidate=get('experimental_late_v.py','_partition_late_v')
    def strip(function):
        function=copy.deepcopy(function);function.name='partition'
        loop=next(n for n in function.body if isinstance(n,ast.For))
        matches=[n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='v' for t in n.targets)]
        assert len(matches)==1;load=matches[0];position=loop.body.index(load);loop.body.remove(load)
        return function,loop,load,position
    a,la,va,pa=strip(original);b,lb,vb,pb=strip(candidate)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    assert ast.dump(va,include_attributes=False)==ast.dump(vb,include_attributes=False)
    assert pa<pb and isinstance(lb.body[pb],ast.Assign) and lb.body[pb].targets[0].id=='acc'


def test_cubin_fingerprint_ignores_debug_and_function_name(monkeypatch):
    import struct,hashlib
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'tools'))
    from cubin_text import fingerprint
    def make(name,debug):
        names=b'\x00.shstrtab\x00.text.'+name.encode()+b'\x00.debug\x00'
        text=b'fixed machine instructions';head=bytearray(64)
        head[:6]=b'\x7fELF\x02\x01';struct.pack_into('<Q',head,40,64)
        struct.pack_into('<HHH',head,58,64,4,1)
        start=64+4*64
        def sh(n,off,size):return struct.pack('<IIQQQQIIQQ',n,1,0,0,off,size,0,0,1,0)
        table=bytes(64)+sh(1,start,len(names))+sh(11,start+len(names),len(text))+sh(names.index(b'.debug'),start+len(names)+len(text),len(debug))
        return bytes(head)+table+names+text+debug
    a=fingerprint(make('original',b'debug1'),'original');b=fingerprint(make('candidate',b'newdebug'),'candidate')
    assert a==b and a['sha256']==hashlib.sha256(b'fixed machine instructions').hexdigest()
    with pytest.raises(ValueError):fingerprint(b'not an ELF','x')
