"""Save cached compiler stages matched by the actual benchmark metadata hash."""
import argparse,hashlib,json,re
from pathlib import Path
from _common import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();targets={}
    for case in ('b1-l8192-d64','b1-l8192-d128'):
        data=json.loads((ROOT/'results/latev-benchmark-v1'/case/'case.json').read_text())
        for variant,c in data['compiled'].items():
            digest=re.search("hash='([^']+)'",c['metadata'])[1]
            targets[digest]={'case':case,'variant':variant,'machine_code':c['machine_code']}
    args.output.mkdir(parents=True,exist_ok=False);found={}
    for path in (ROOT/'.cache/triton').rglob('*.json'):
        try:data=json.loads(path.read_text())
        except (ValueError,OSError):continue
        digest=data.get('hash') if isinstance(data,dict) else None
        if digest not in targets:continue
        row={**targets[digest],'compiler_hash':digest,'files':{},'missing':[]}
        for suffix in ('.ttir','.ttgir','.llir','.ptx'):
            source=path.with_suffix(suffix)
            if not source.is_file():row['missing'].append(suffix);continue
            content=source.read_bytes();target=args.output/row['case']/(row['variant']+suffix)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
            row['files'][target.relative_to(args.output).as_posix()]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
        found[digest]=row
    if set(found)!=set(targets):raise RuntimeError('one or more compiled cache entries are missing')
    (args.output/'manifest.json').write_bytes((json.dumps({'scope':'Compiler text snapshots; debug information is retained','entries':list(found.values())},indent=2)+'\n').encode())
    print('Saved compiler stages for',len(found),'compiled kernels')


if __name__=='__main__':main()
