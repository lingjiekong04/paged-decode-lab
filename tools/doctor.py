import argparse
import json
from _common import fingerprint, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read-only environment inspection; no installations or downloads")
    parser.add_argument("--output")
    args = parser.parse_args()
    data = fingerprint()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, data)
