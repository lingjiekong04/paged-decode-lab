"""Check delivered upstream identity and exercise config contracts without GPU deps."""
import ast
import hashlib
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_vendor_matches_recorded_manifest():
    manifest = json.loads((ROOT / "upstream.json").read_text())
    assert manifest["commit"] == "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
    for entry in manifest["files"]:
        data = (ROOT / "vendor/nano-vllm" / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["delivered_sha256"], entry["path"]
    assert (ROOT / "vendor/nano-vllm/LICENSE").is_file()


def test_every_delivered_python_file_parses():
    for folder in ("src", "tools", "tests", "vendor/nano-vllm"):
        for path in (ROOT / folder).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_sampling_parameters_are_real_upstream_definitions():
    source = ROOT / "vendor/nano-vllm/nanovllm/sampling_params.py"
    namespace = {"__name__": "_test_sampling"}
    # Dataclasses need the containing module registered even in isolated execution.
    import sys
    module = types.ModuleType(namespace["__name__"])
    sys.modules[module.__name__] = module
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    cls = module.SamplingParams
    assert cls(temperature=0).temperature == 0
    assert cls(temperature=0.7).temperature == 0.7
    for temperature in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cls(temperature=temperature)
    with pytest.raises(ValueError):
        cls(max_tokens=0)


def test_config_declares_public_backend_and_fixed_pool():
    tree = ast.parse((ROOT / "vendor/nano-vllm/nanovllm/config.py").read_text())
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config")
    fields = {n.target.id for n in config.body if isinstance(n, ast.AnnAssign)}
    assert {"decode_backend", "kv_cache_blocks"} <= fields


def test_graph_buckets_cover_nonstandard_maximum():
    # Execute the actual assignment from the delivered runner, not a copied formula.
    tree = ast.parse((ROOT / "vendor/nano-vllm/nanovllm/engine/model_runner.py").read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Attribute) and t.attr == "graph_bs" for t in n.targets))
    expression = compile(ast.Expression(assignment.value), "graph_buckets", "eval")
    for max_bs in (1, 3, 7, 17, 31, 32, 65, 512):
        buckets = eval(expression, {"max_bs": max_bs})
        assert buckets[-1] == max_bs
        assert min(buckets) >= 1
        assert all(any(bucket >= batch for bucket in buckets) for batch in range(1, max_bs + 1))
