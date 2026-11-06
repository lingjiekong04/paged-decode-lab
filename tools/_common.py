from pathlib import Path
import importlib.metadata
import json
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/nano-vllm")]


def write_json(path, data):
    path = Path(path)
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def fingerprint():
    import datetime
    import hashlib
    import os
    result = {"python": sys.version, "platform": platform.platform(), "packages": {},
              "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "src/paged_decode").glob("*.py"))}}
    for name in ("torch", "triton", "flash-attn", "transformers", "numpy", "pytest", "paged-decode-lab"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["gpu"] = {"name": props.name, "memory_bytes": props.total_memory,
                             "compute_capability": [props.major, props.minor]}
    except ImportError:
        result["cuda_available"] = False
    import subprocess
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        result["nvidia_smi"] = proc.stdout.strip() if proc.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        result["nvidia_smi"] = "unavailable"
    return result


def require_gpu():
    if sys.platform != "linux":
        raise RuntimeError("GPU tools require Linux; use tools/check.py for the CPU suite")
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("pinned nano-vLLM requires Python 3.10–3.12; recommended 3.11")
    import torch
    import triton  # noqa: F401
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("a CUDA device with BF16 support is required")
    return torch


def summarize(values):
    values = [float(v) for v in values]
    if not values:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    # Nearest-rank p95, also valid for short repeat lists.
    import math
    return {"count": len(values), "median": statistics.median(values),
            "p95": ordered[math.ceil(.95 * len(values)) - 1],
            "min": min(values), "max": max(values)}


def model_identity(model):
    """Require the manifest written by download_model.py; hash small config files."""
    import hashlib
    model = Path(model).resolve()
    manifest_path = model / "paged_decode_model.json"
    if not manifest_path.is_file():
        raise ValueError("model needs paged_decode_model.json; use download_model.py (can reuse cached weights)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configured = json.loads((ROOT / "configs/models.json").read_text())
    legacy = ROOT / "configs/models-legacy.json"
    if legacy.exists():
        configured.update({"legacy_" + k: v for k, v in json.loads(legacy.read_text()).items()})
    expected = next((m for m in configured.values() if m["repo_id"] == manifest["repo_id"]), None)
    if expected is None or expected["revision"] != manifest["revision"]:
        raise ValueError("model revision does not match configs/models.json")
    if manifest["weights"] != expected["weights"]:
        raise ValueError("weight manifest differs from pinned official SHA256 values")
    for name, digest in manifest["files"].items():
        path = model / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"model configuration/tokenizer changed: {name}")
    for entry in manifest["weights"]:
        path = model / entry["name"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"model weight file missing or size changed: {entry['name']}")
    return manifest
