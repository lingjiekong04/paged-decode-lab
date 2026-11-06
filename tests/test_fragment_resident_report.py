import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def report():
    path=Path(__file__).resolve().parents[1]/'tools/report_fragment_resident.py'
    spec=importlib.util.spec_from_file_location('resident_report',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def operator_data():
    rows=[dict(input_tokens=n,context=n+16,variant=v,stage=s,parts=g,unroll=u,repeat=r,
               us_per_call={'original':10.,'shared_s3':12.,'shared_s2':6.}[v])
          for n in (512,2048) for v in ('original','shared_s3','shared_s2')
          for s,g in [('partition',x) for x in (18,24,27,28,32,33,36)]+[('full',36)]
          for u in (32,128) for r in range(5)]
    resources=[dict(input_tokens=n,variant=v,driver_max_active_blocks_per_sm=2 if v!='shared_s3' else 1,
                    dynamic_shared_bytes=77952 if v=='shared_s2' else 79232)
               for n in (512,2048) for v in ('original','shared_s3','shared_s2')]
    return dict(status='passed',rows=rows,resources=resources,real_tensor_native_bytes_equal=True)


def model_data():
    rows=[dict(input_tokens=n,backend=b,mode=m,repeat=r,wall_seconds=w,output_tokens=64,
               output_tokens_per_second=64/w,decode_event_ms=[1.]*63,peak_allocated_bytes=100,tokens_equal_native=True)
          for n in (512,2048) for b,w in [('native',2.),('original',4.),('shared_s3',3.),('shared_s2',1.)]
          for m in ('eager','graph') for r in range(5)]
    validation=[dict(input_tokens=n,backend=b,mode=m,free_tokens_equal=True,
                     steps=[dict(step=s,bytes_equal=True,max_abs=0.) for s in range(64)])
                for n in (512,2048) for b in ('native','original','shared_s3','shared_s2') for m in ('eager','graph')]
    return dict(status='passed',rows=rows,validation=validation,resources={'shared_s2':{'driver_max_active_blocks_per_sm':2}})


def test_operator_pairing_and_full_stage(report):
    groups=report.operator_summary(operator_data())
    assert len(groups)==16
    assert sum(g['stage']=='full' for g in groups)==2
    assert all(g['over_s2']['shared_s3']['median']==2 for g in groups)
    assert all(g['over_s2']['original']['median']==pytest.approx(10/6) for g in groups)


@pytest.mark.parametrize('failure',['missing','duplicate','nonfinite','nonresident','not_reduced','context','numerical'])
def test_bad_operator_evidence(report,failure):
    d=operator_data()
    if failure=='missing':d['rows'].pop()
    if failure=='duplicate':d['rows'].append(d['rows'][0])
    if failure=='nonfinite':d['rows'][0]['us_per_call']=float('inf')
    if failure=='nonresident':d['resources'][2]['driver_max_active_blocks_per_sm']=1
    if failure=='not_reduced':d['resources'][2]['dynamic_shared_bytes']=79232
    if failure=='context':d['rows'][0]['context']=512
    if failure=='numerical':d['real_tensor_native_bytes_equal']=False
    with pytest.raises(ValueError):report.operator_summary(d)


def test_model_pairing_and_units(report):
    rows=report.model_summary(model_data());assert len(rows)==4
    for r in rows:
        assert r['wall_over_s2']['shared_s3']['median']==3
        assert r['wall_over_s2']['original']['median']==4
        assert r['variants']['shared_s2']['tps']['median']==64


@pytest.mark.parametrize('failure',['missing','duplicate','nonfinite','event','tps','greedy','logits','validation','nonresident'])
def test_bad_model_evidence(report,failure):
    d=model_data()
    if failure=='missing':d['rows'].pop()
    if failure=='duplicate':d['rows'].append(d['rows'][0])
    if failure=='nonfinite':d['rows'][0]['wall_seconds']=float('nan')
    if failure=='event':d['rows'][0]['decode_event_ms'][0]=0.
    if failure=='tps':d['rows'][0]['output_tokens_per_second']=1.
    if failure=='greedy':d['validation'][0]['free_tokens_equal']=False
    if failure=='logits':d['validation'][0]['steps'][0]['bytes_equal']=False
    if failure=='validation':d['validation'].pop()
    if failure=='nonresident':d['resources']['shared_s2']['driver_max_active_blocks_per_sm']=1
    with pytest.raises(ValueError):report.model_summary(d)
