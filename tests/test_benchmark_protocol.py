import pytest
from paged_decode.benchmark_protocol import interval_per_operator,convergence_delta


def test_graph_nodes_are_counted_as_logical_operators():
    # 1280 operators in 1.28ms is 1us/operator, not 128us/replay.
    assert interval_per_operator(1.28,10,128)==1.
    assert interval_per_operator(1.28,1280,1)==1.
    assert convergence_delta(10.4,10.)==pytest.approx(.04)


@pytest.mark.parametrize('args',[(0.,10,128),(float('nan'),10,128),(1.,0,128),(1.,10,0),(1.,10,True)])
def test_invalid_denominators_cannot_produce_latency(args):
    with pytest.raises(ValueError):interval_per_operator(*args)


def test_busy_gpu_is_rejected_before_cuda_initialization(monkeypatch,tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    import benchmark_operator_v2
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    def query(cmd,**kwargs):
        return SimpleNamespace(returncode=0,stderr='',stdout='0, GPU-test\n' if '--query-gpu=index,uuid' in cmd else '12345, GPU-test\n')
    def forbidden():raise AssertionError('must not initialize CUDA on a busy GPU')
    monkeypatch.setattr(benchmark_operator_v2.subprocess,'run',query)
    monkeypatch.setattr(benchmark_operator_v2,'require_gpu',forbidden)
    target=tmp_path/'unused'
    monkeypatch.setattr('sys.argv',['benchmark_operator_v2.py','--geometry','small','--output',str(target)])
    with pytest.raises(RuntimeError,match='resident compute processes'):benchmark_operator_v2.main()
    assert not target.exists()
