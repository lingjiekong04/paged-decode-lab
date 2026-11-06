"""Export selected project results; preserve hashes for large reference tensors.

Only project results are readable. Model weights, environments and credentials are
never included. Default export omits .pt payloads but indexes their size/SHA256.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import zipfile
from _common import ROOT


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include", nargs="+", required=True, help="paths relative to results/")
    parser.add_argument("--with-tensors", action="store_true")
    parser.add_argument("--index-name", default="EVIDENCE_INDEX.json", help="JSON basename; use a separate index for later phases")
    args = parser.parse_args()
    if Path(args.index_name).name != args.index_name or not args.index_name.endswith(".json"):
        parser.error("index-name must be a JSON basename")
    base = (ROOT / "results").resolve()
    target = Path(args.output).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("export destination must stay inside the project")
    paths = set()
    for name in args.include:
        candidate = (base / name).resolve()
        if not candidate.is_relative_to(base) or not candidate.exists():
            raise ValueError(f"invalid/missing result path: {name}")
        paths.update(candidate.rglob("*") if candidate.is_dir() else [candidate])
    manifest, tensors = [], []
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if not path.is_file() or path == target or path.suffix == ".zip":
                continue
            if not path.resolve().is_relative_to(base):
                raise ValueError("result symlink leaves project results")
            entry = {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                     "sha256": digest(path)}
            if path.suffix == ".pt" and not args.with_tensors:
                tensors.append(entry)
                continue
            archive.write(path, entry["path"])
            manifest.append(entry)
        archive.writestr("results/" + args.index_name, json.dumps({
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "files": manifest, "omitted_reference_tensors": tensors,
            "tensor_note": "reference tensors retained in the same relative paths on the execution server",
        }, indent=2) + "\n")
    print(json.dumps({"archive": str(target), "files": len(manifest), "omitted_tensors": len(tensors),
                      "archive_sha256": digest(target)}))


if __name__ == "__main__":
    main()
