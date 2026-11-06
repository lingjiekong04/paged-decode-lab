"""Export immutable residency evidence, including tensors; never launch GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();files={}
    def add(path):
        if path.is_file() and '__pycache__' not in path.parts:
            files[path.relative_to(root).as_posix()]=path.read_bytes()
    for tag in ('residency-resources-v1','residency-snapshots-v1','residency-snapshots-v2',
                'residency-padding-smoke-v1','residency-benchmark-v1','residency-benchmark-v3',
                'residency-host-ncu-v1','residency-probe-sources'):
        directory=root/'results'/tag
        if directory.is_dir():
            for path in sorted(directory.rglob('*')):add(path)
        for suffix in ('.log','.job.json','.exit'):
            add(root/'results'/(tag+suffix))
    for tag in ('residency-ncu-baseline-v1','residency-ncu-direct-baseline-v1'):
        for suffix in ('.log','.job.json','.csv','.ncu-rep','.target.json'):
            add(root/'results'/(tag+suffix))
    for name in ('residency-cpu-local.log','residency-cpu-remote.log'):
        add(root/'results'/name)
    for tag in ('residency-snapshots-v2','residency-benchmark-v3'):
        directory=root/'results'/tag
        monitor=json.loads((directory/'gpu-monitor.json').read_text())
        if monitor['violations'] or not monitor['rows']:raise ValueError('invalid GPU monitoring')
        config=json.loads((directory/'arguments.json').read_text())
        for name,digest in config['source_sha256'].items():
            saved=directory/'sources'/name
            if hashlib.sha256(saved.read_bytes()).hexdigest()!=digest:raise ValueError(name)
            if saved.read_bytes()!=(root/name).read_bytes():raise ValueError('source changed: '+name)
    summary=json.loads(files['results/residency-benchmark-v3/summary.json'])
    if summary['status']!='passed' or len(summary['rows'])!=420:raise ValueError('incomplete matrix')
    snapshots=json.loads(files['results/residency-snapshots-v2/snapshots.json'])
    for row in snapshots['rows']:
        data=files['results/residency-snapshots-v2/'+row['file']]
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('snapshot hash mismatch')
    index={'phase':'D256 partition residency diagnosis',
           'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}}
    files['results/RESIDENCY_EVIDENCE_INDEX.json']=(json.dumps(index,indent=2)+'\n').encode()
    with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):archive.writestr(name,data)
    print(json.dumps({'files':len(files),'bytes':args.output.stat().st_size,
                      'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
