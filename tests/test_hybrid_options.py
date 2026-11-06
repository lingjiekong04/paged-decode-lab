"""CPU-only public options and CLI behavior, including pre-load failures."""
from pathlib import Path
import pytest
from paged_decode.hybrid_options import HYBRID_BACKENDS, validate_hybrid_options
from paged_decode.spec import BACKENDS, validate_engine_options


def test_matched_is_scoped_to_fixed_hybrid_runner():
    assert 'triton_matched' in HYBRID_BACKENDS
    assert 'triton_matched' not in BACKENDS
    with pytest.raises(ValueError,match='unknown decode_backend'):
        validate_engine_options('triton_matched',1,256,1,1024)


@pytest.mark.parametrize('options',[
    ('unknown','eager',1,1024),('triton_matched','unknown',1,1024),
    ('triton_matched','eager',0,1024),('triton_matched','eager',True,1024),
    ('triton_matched','eager',1,0),('triton_matched','graph',1,2)])
def test_invalid_runner_options(options):
    with pytest.raises(ValueError): validate_hybrid_options(*options)


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    import qwen35
    return qwen35


def test_cli_defaults_and_fixed_capacity(cli):
    base=['--model','unused','--output','unused']
    assert cli.parse_args(base).batch==1
    assert cli.parse_args(base+['--task','benchmark']).batch==8
    args=cli.parse_args(base+['--backend','triton_matched','--mode','graph','--batch','8','--max-context','2176'])
    assert (args.backend,args.batch,args.max_context)==('triton_matched',8,2176)


@pytest.mark.parametrize('extra',[
    ['--task','validate','--batch','8'],['--max-context','0'],
    ['--task','benchmark','--max-context','512'],['--backend','typo'],
    ['--backend','triton_matched','--task','validate']])
def test_cli_rejects_unsupported_requests_before_loading(cli,extra):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(['--model','unused','--output','unused',*extra])
    assert exc.value.code==2


@pytest.mark.parametrize('extra',[['--batch','8'],['--tokens','0'],['--batch','0']])
def test_suite_rejects_invalid_configuration_before_creating_outputs(cli,monkeypatch,tmp_path,extra):
    import run_qwen35_suite
    target=tmp_path/'unused'
    monkeypatch.setattr('sys.argv',['run_qwen35_suite.py','--model','unused','--output',str(target),
                                  '--task','validate','--include-matched',*extra])
    with pytest.raises(SystemExit) as exc: run_qwen35_suite.main()
    assert exc.value.code==2 and not target.exists()
