"""Rebuild pinned nano-vLLM from a hash-verified archive and exact overlays."""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rebuild(archive, destination):
    manifest = json.loads((ROOT / "upstream.json").read_text(encoding="utf-8"))
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Pinned archive SHA256 mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nano-rebuild-", dir=destination.parent) as tmp:
        target = Path(tmp) / "nano-vllm"
        target.mkdir()
        with zipfile.ZipFile(archive) as z:
            roots = {Path(n).parts[0] for n in z.namelist() if n}
            if len(roots) != 1:
                raise ValueError("Unexpected archive root structure")
            prefix = next(iter(roots))
            for item in manifest["files"]:
                rel = Path(item["path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("Unsafe upstream manifest path")
                data = z.read(prefix + "/" + rel.as_posix())
                if hashlib.sha256(data).hexdigest() != item["upstream_sha256"]:
                    raise ValueError(f"Upstream file mismatch: {rel}")
                if item["modified"]:
                    data = (ROOT / "overlays/nano-vllm" / rel).read_bytes()
                if hashlib.sha256(data).hexdigest() != item["delivered_sha256"]:
                    raise ValueError(f"Overlay mismatch: {rel}")
                p = target / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        target.rename(destination)
    print(f"Verified {len(manifest['files'])} files: {destination}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, help="Use an existing pinned ZIP offline")
    p.add_argument("--destination", type=Path, default=ROOT / "vendor/nano-vllm")
    args = p.parse_args()
    if args.destination.exists():
        raise FileExistsError("Destination exists; inspect/reuse it instead of overwriting")
    if args.archive:
        rebuild(args.archive, args.destination)
    else:
        manifest = json.loads((ROOT / "upstream.json").read_text())
        with tempfile.TemporaryDirectory(prefix="nano-download-") as tmp:
            archive = Path(tmp) / "baseline.zip"
            with urllib.request.urlopen(manifest["archive_url"], timeout=60) as r:
                with archive.open("xb") as f:
                    shutil.copyfileobj(r, f)
            rebuild(archive, args.destination)


if __name__ == "__main__":
    main()
