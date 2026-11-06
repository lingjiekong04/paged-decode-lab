"""Transport can change only if the exact pinned weight digest is preserved."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_downloader():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("_download_test", ROOT / "tools/accelerate_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(monkeypatch, tmp_path, *, corrupt=False, bad_range=False):
    module = load_downloader()
    payload = b"0123456789abcdef" * 150000
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (tmp_path / "configs").mkdir()
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "configs/models.json").write_text(json.dumps({"small": {
        "repo_id": "test/model", "revision": "test-revision", "weights": [
            {"name": "model.safetensors", "bytes": len(payload), "sha256": digest}]}}))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["accelerate_weights.py", "--size", "small", "--output", str(model),
                                      "--workers", "2", "--chunk-mib", "1"])
    calls = []

    class Response(io.BytesIO):
        def __init__(self, data, start, end):
            super().__init__(data)
            self.status = 200 if bad_range else 206
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}

    def open_url(request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        calls.append((start, end))
        data = payload[start:end + 1]
        if corrupt:
            data = b"X" + data[1:]
        return Response(data, start, end)

    monkeypatch.setattr(module.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    # Windows CPU CI has no os.pwrite; emulate positional writes with a lock.
    if not hasattr(os, "pwrite"):
        write_lock = threading.Lock()

        def pwrite(fd, data, offset):
            with write_lock:
                os.lseek(fd, offset, os.SEEK_SET)
                return os.write(fd, data)
        monkeypatch.setattr(os, "pwrite", pwrite, raising=False)
    return module, model, payload, calls


def test_exact_weight_promoted_after_parallel_ranges(monkeypatch, tmp_path):
    module, model, payload, calls = configure(monkeypatch, tmp_path)
    module.main()
    assert (model / "model.safetensors").read_bytes() == payload
    manifest = json.loads((model / "paged_decode_model.json").read_text())
    assert manifest["weights"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(calls) == 3


@pytest.mark.parametrize("corrupt,bad_range", [(True, False), (False, True)])
def test_bad_transport_never_promotes_weight(monkeypatch, tmp_path, corrupt, bad_range):
    module, model, _, _ = configure(monkeypatch, tmp_path, corrupt=corrupt, bad_range=bad_range)
    with pytest.raises(RuntimeError):
        module.main()
    assert not (model / "model.safetensors").exists()
    assert not (model / "paged_decode_model.json").exists()
