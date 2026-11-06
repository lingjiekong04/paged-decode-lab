import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def aggregate(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'tools'))
    s=importlib.util.spec_from_file_location('model_perf_report',root/'tools/report_fragment_model_performance.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.aggregate


def records():
    return [dict(input_tokens=n,mode=m,backend=b,repeat=i,wall_seconds=w,output_tokens_per_second=64/w,
        decode_event_ms=[1.]*63,decode_event_median_ms=1.,peak_allocated_bytes=100,tokens_equal_native=True)
        for n in (512,2048) for m in ('eager','graph') for b,w in [('native',2.),('original',4.),('shared',1.)] for i in range(5)]


def test_paired_ratios_and_units(aggregate):
    result=aggregate(records());assert len(result)==4
    for r in result:
        assert r['ratios']['original']['wall_paired_median']==4
        assert r['ratios']['native']['wall_paired_median']==2
        assert r['variants']['shared']['tps_median']==64


def test_missing_or_duplicate_rounds_rejected(aggregate):
    a=records()
    with pytest.raises(ValueError):aggregate(a[:-1])
    a[-1]['repeat']=3
    with pytest.raises(ValueError):aggregate(a)


@pytest.mark.parametrize('field,value',[('tokens_equal_native',False),('wall_seconds',float('inf')),('decode_event_ms',[0.]*63)])
def test_invalid_observations_rejected(aggregate,field,value):
    a=records();a[0][field]=value
    with pytest.raises(ValueError):aggregate(a)
