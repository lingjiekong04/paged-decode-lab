"""Export seven completed resident-pipeline campaigns and their exact checkpoints."""
import argparse,hashlib,json,zipfile
from pathlib import Path

TAGS=('smem2-smoke-v1','smem2-correct-v1','smem2-operator-v1','smem2-model-small-v1',
      'smem2-model-large-v1','smem2-perf-small-v1','smem2-perf-large-v1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.root.resolve();files={}
    for tag in TAGS:
        receipt_path=root/'results'/(tag+'.job.json');receipt=json.loads(receipt_path.read_text())
        if receipt['exit_code']!=0:raise ValueError('experiment failed: '+tag)
        files[receipt_path.relative_to(root).as_posix()]=receipt_path.read_bytes()
        archive=root/'results'/(tag+'.checkpoint.zip');data=archive.read_bytes()
        if hashlib.sha256(data).hexdigest()!=receipt['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
        files[archive.relative_to(root).as_posix()]=data
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('invalid checkpoint')
            for name in z.namelist():
                path=(root/'results'/tag/name).resolve()
                if not path.is_relative_to(root/'results'/tag):raise ValueError('unsafe checkpoint path')
                files[path.relative_to(root).as_posix()]=z.read(name)
        monitor=json.loads(files[f'results/{tag}/gpu-monitor.json'])
        if monitor['violations'] or not monitor['rows']:raise ValueError('GPU monitor failed')
        if json.loads(files[f'results/{tag}/summary.json'])['status']!='passed':raise ValueError('campaign incomplete')
        for suffix in ('.log',):
            log=root/'results'/(tag+suffix)
            files[log.relative_to(root).as_posix()]=log.read_bytes()
        for meta_name in ('arguments.json','resident-experiment.json'):
            meta=files.get(f'results/{tag}/{meta_name}')
            if meta is None:continue
            for name,digest in json.loads(meta).get('source_sha256',{}).items():
                saved=files.get(f'results/{tag}/sources/{name}')
                if saved is None or hashlib.sha256(saved).hexdigest()!=digest:raise ValueError('source snapshot mismatch: '+name)
                if saved!=(root/name).read_bytes():raise ValueError('current source differs: '+name)
    for name in ('smem2-chain-small-v1.log','smem2-chain-large-v1.log','smem2-cpu-local.log','smem2-cpu-remote.log'):
        path=root/'results'/name
        if path.exists():files[path.relative_to(root).as_posix()]=path.read_bytes()
    index={'phase':'D256 resident two-stage pipeline','files':{n:hashlib.sha256(b).hexdigest() for n,b in sorted(files.items())}}
    files['results/SMEM2_EVIDENCE_INDEX.json']=(json.dumps(index,indent=2)+'\n').encode()
    with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):z.writestr(name,data)
    print(json.dumps({'files':len(files),'bytes':args.output.stat().st_size,'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
