import argparse
import hashlib
import json
from pathlib import Path
from _common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description="Explicitly download/reuse one pinned public model snapshot")
    parser.add_argument("--size", choices=tuple(json.loads((ROOT / "configs/models.json").read_text())), default="small")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / "configs/models.json").read_text())[args.size]
    destination = Path(args.output).resolve()
    manifest = destination / "paged_decode_model.json"
    if manifest.exists():
        raise FileExistsError(f"model manifest exists: {manifest}")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"], local_dir=str(destination))
    files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        path = destination / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "config.json" not in files:
        raise RuntimeError("download incomplete: missing model config")
    weights = []
    for path in sorted(destination.glob("*.safetensors")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        weights.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    if not weights:
        raise RuntimeError("download incomplete: no safetensors weights")
    if weights != spec["weights"]:
        raise RuntimeError("weights do not match the pinned official size/SHA256 manifest")
    write_json(manifest, {"repo_id": spec["repo_id"], "revision": spec["revision"], "files": files, "weights": weights,
                          "note": "weight hashes recorded on download; benchmark rechecks sizes and small files"})
    print(manifest)


if __name__ == "__main__":
    main()
