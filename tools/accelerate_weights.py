"""Range-download weights from ModelScope and verify the pinned HF SHA256.

Only transport changes: configuration/tokenizer files still come from the pinned
HF snapshot. Files are promoted from .partial only after exact SHA256 validation.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
from _common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=tuple(json.loads((ROOT / "configs/models.json").read_text())), required=True)
    parser.add_argument("--output", required=True, help="existing pinned snapshot directory with configs")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--file-workers", type=int, default=1, help="parallel shards; total range workers <=64")
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers * args.file_workers <= 64 or min(args.workers, args.file_workers, args.chunk_mib) <= 0:
        parser.error("positive worker/chunk counts required; total workers must not exceed 64")
    spec = json.loads((ROOT / "configs/models.json").read_text())[args.size]
    destination = Path(args.output)
    if not (destination / "config.json").is_file():
        raise ValueError("fetch pinned config/tokenizer files using download_model.py first")
    if (destination / "paged_decode_model.json").exists():
        raise FileExistsError("model manifest already exists")
    def download_one(entry):
        name, total, digest = entry["name"], entry["bytes"], entry["sha256"]
        final = destination / name
        if final.exists():
            actual = hashlib.sha256()
            with final.open("rb") as stream:
                for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    actual.update(data)
            if final.stat().st_size != total or actual.hexdigest() != digest:
                raise ValueError(f"existing weight differs from pinned SHA256: {final}")
            return entry
        partial = destination / (name + ".ranged.partial")
        checkpoint = destination / (name + ".ranges.json")
        chunk_size = args.chunk_mib * 1024 * 1024
        count = (total + chunk_size - 1) // chunk_size
        done = set()
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if state["sha256"] != digest or state["chunk_size"] != chunk_size or not partial.exists():
                raise ValueError("resume checkpoint identity differs")
            done = set(state["done"])
        fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o600)
        os.ftruncate(fd, total)
        lock = threading.Lock()
        url = f"https://modelscope.cn/models/{spec['repo_id']}/resolve/master/{name}"
        pinned_fallback = f"https://hf-mirror.com/{spec['repo_id']}/resolve/{spec['revision']}/{name}"

        def fetch(index):
            start = index * chunk_size
            end = min(total, start + chunk_size) - 1
            for attempt in range(8):
                try:
                    source = url if attempt % 2 == 0 else pinned_fallback
                    request = urllib.request.Request(source, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        expected_range = f"bytes {start}-{end}/{total}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                            raise RuntimeError(f"range response differs: {response.status}, {response.headers.get('Content-Range')}")
                        cursor = start
                        while data := response.read(1024 * 1024):
                            if cursor + len(data) > end + 1:
                                raise RuntimeError("range response exceeds requested size")
                            written = 0
                            while written < len(data):
                                written += os.pwrite(fd, data[written:], cursor + written)
                            cursor += len(data)
                        if cursor != end + 1:
                            raise RuntimeError("truncated range")
                    with lock:
                        done.add(index)
                        temp = checkpoint.with_suffix(".tmp")
                        temp.write_text(json.dumps({"sha256": digest, "chunk_size": chunk_size,
                                                    "done": sorted(done)}))
                        temp.replace(checkpoint)
                        if len(done) % 16 == 0 or len(done) == count:
                            print(name, f"{len(done)}/{count} chunks", flush=True)
                    return
                except Exception:
                    if attempt == 7:
                        raise
                    time.sleep(min(2 ** attempt, 10))
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(fetch, (i for i in range(count) if i not in done)))
            os.fsync(fd)
        finally:
            os.close(fd)
        actual = hashlib.sha256()
        with partial.open("rb") as stream:
            for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                actual.update(data)
        if actual.hexdigest() != digest:
            raise RuntimeError(f"SHA256 differs from pinned HF revision: {name}")
        partial.rename(final)
        print("VERIFIED", name, digest, flush=True)
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        all_weights = list(pool.map(download_one, spec["weights"]))
    files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        path = destination / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(destination / "paged_decode_model.json", {
        "repo_id": spec["repo_id"], "revision": spec["revision"], "files": files, "weights": all_weights,
        "transport": "ModelScope ranges; exact pinned Hugging Face weight SHA256 verified"})


if __name__ == "__main__":
    main()
