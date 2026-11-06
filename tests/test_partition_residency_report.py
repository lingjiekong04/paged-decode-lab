import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def reducer():
    path=Path(__file__).resolve().parents[1]/'tools/report_partition_residency.py'
    spec=importlib.util.spec_from_file_location('residency_report',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.aggregate


def data():
    rows=[dict(input_tokens=n,context=n+16,variant=v,grid_parts=g,unroll=u,repeat=r,
               total_ctas=g*4,active_ctas=4*min(g,(n+79)//64),
               us_per_call={'baseline':10.,'baseline_padded':15.,'shared':12.}[v])
          for n in (512,2048) for v in ('baseline','baseline_padded','shared')
          for g in (18,24,27,28,32,33,36) for u in (32,128) for r in range(5)]
    resources={v:dict(registers_per_thread=255,dynamic_shared_bytes=83584 if v=='baseline_padded' else 79232,
                      static_shared_bytes=4352 if v=='shared' else 0)
               for v in ('baseline','baseline_padded','shared')}
    return dict(status='passed',rows=rows,checks=[dict(input_tokens=n,native_bytes_equal=True,
                 padded_partials_bytes_equal=True,resources=resources) for n in (512,2048)])


def test_paired_padding_and_grid(reducer):
    groups=reducer(data())
    assert len(groups)==14
    long=next(g for g in groups if g['context']==2064 and g['parts']==36)
    assert long['active_ctas']==132
    assert long['total_ctas']==144
    assert long['ratios']['padded_over_baseline']['median']==1.5
    assert long['ratios']['baseline_over_shared']['median']==pytest.approx(10/12)
    assert all(v['converged_3pct'] for v in long['variants'].values())


@pytest.mark.parametrize('error',['missing','duplicate','nonfinite','active','padding','registers','numerical'])
def test_invalid_controls_rejected(reducer,error):
    d=data()
    if error=='missing':d['rows'].pop()
    if error=='duplicate':d['rows'].append(d['rows'][0])
    if error=='nonfinite':d['rows'][0]['us_per_call']=float('nan')
    if error=='active':d['rows'][0]['active_ctas']=144
    if error=='padding':d['checks'][0]['resources']['baseline_padded']['dynamic_shared_bytes']=79232
    if error=='registers':d['checks'][0]['resources']['shared']['registers_per_thread']=128
    if error=='numerical':d['checks'][0]['padded_partials_bytes_equal']=False
    with pytest.raises(ValueError):reducer(d)


@pytest.mark.parametrize('case,expected',[
    ('timeout','timeout_no_valid_counters'),('permission','counter_permission_denied'),
    ('empty','completed_without_metrics'),('metrics','completed')])
def test_profiler_outcome_requires_evidence(tmp_path,case,expected):
    import json
    path=Path(__file__).resolve().parents[1]/'tools/report_partition_residency.py'
    spec=importlib.util.spec_from_file_location('ncu_result_report',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tag='residency-ncu-test-v1'
    (tmp_path/(tag+'.job.json')).write_text(json.dumps(dict(tag=tag,timeout=case in ('timeout','permission'),exit_code=-15 if case in ('timeout','permission') else 0)))
    (tmp_path/(tag+'.log')).write_text('ERR_NVGPUCTRPERM' if case=='permission' else '')
    (tmp_path/(tag+'.csv')).write_text('"Metric Name","Metric Value"\n"Achieved Occupancy","5.0"\n' if case=='metrics' else '==WARNING== No kernels were profiled.\n')
    result=module.ncu_status(tmp_path)
    assert result[0]['status']==expected
