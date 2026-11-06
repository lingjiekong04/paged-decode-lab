import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def reducer():
    source=Path(__file__).resolve().parents[1]/'tools/report_fragment_model_stages.py'
    spec=importlib.util.spec_from_file_location('stage_report',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.aggregate


def data():
    rows=[]
    for n in (512,2048):
        for b in ('native','original','shared'):
            for tagged in (False,True):
                for r in range(5):
                    stages={'forward':9.,'module.3':3.,'operator.3':1.} if tagged else None
                    rows.append(dict(input_tokens=n,backend=b,instrumented=tagged,repeat=r,
                        steps=[dict(step=s,context=n+s,outer_ms=11. if tagged else 10.,
                                    logits_bytes_equal=True,stages_ms=stages) for s in range(1,17)]))
    micro=[dict(input_tokens=n,context=n+16,layer=3,backend=b,stage=stage,repeat=r,
                us_per_call=2. if b=='original' else 1.,unroll=64,replays=20)
           for n in (512,2048) for b in ('native','original','shared')
           for stage in (('full',) if b=='native' else ('full','partition','merge')) for r in range(5)]
    return dict(status='passed',rows=rows,micro=micro,metadata=[dict(input_tokens=n,context=n+16,layer=3,
            all_outputs_bytes_equal_native=True) for n in (512,2048)])


def test_paired_ratios_and_nested_fraction(reducer):
    output=reducer(data())
    assert len(output)==2
    for group in output:
        assert group['isolated_original_over_shared']['partition']['median']==2.
        assert group['variants']['original']['tagged_operator_fraction']['median']==pytest.approx(1/9)
        assert group['variants']['shared']['tagging_overhead_fraction']['median']==pytest.approx(.1)


@pytest.mark.parametrize('failure',['missing','duplicate','nonfinite','divergence','nesting','micro','snapshot'])
def test_invalid_evidence_rejected(reducer,failure):
    d=data()
    if failure=='missing':d['rows'].pop()
    if failure=='duplicate':d['rows'].append(d['rows'][0])
    if failure=='nonfinite':d['rows'][0]['steps'][0]['outer_ms']=float('nan')
    if failure=='divergence':d['rows'][0]['steps'][0]['logits_bytes_equal']=False
    if failure=='nesting':d['rows'][5]['steps'][0]['stages_ms']['operator.3']=4.
    if failure=='micro':d['micro'].pop()
    if failure=='snapshot':d['metadata'][0]['all_outputs_bytes_equal_native']=False
    with pytest.raises(ValueError):reducer(d)
