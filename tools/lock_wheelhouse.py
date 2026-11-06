"""Create a hash-locked offline artifact list; installed dependencies still need pip check."""
import argparse,email,hashlib,json,zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[]
    for path in sorted(args.directory.glob('*.whl')):
        with zipfile.ZipFile(path) as z:
            metadata=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        rows.append({'filename':path.name,'name':metadata['Name'],'version':metadata['Version'],
            'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    if not rows:raise ValueError('no wheels found')
    names=[r['name'].lower().replace('_','-') for r in rows]
    if len(set(names))!=len(names):raise ValueError('multiple candidates for a distribution; curate before locking')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    lines=['# Run from the parent of wheelhouse; exact offline artifacts.']
    lines += [f"./{args.directory.name}/{r['filename']} --hash=sha256:{r['sha256']}" for r in rows]
    args.output.write_bytes(('\n'.join(lines)+'\n').encode())
    args.output.with_suffix('.json').write_bytes((json.dumps({'wheels':rows},indent=2)+'\n').encode())
    print('Locked',len(rows),'wheel artifacts')


if __name__=='__main__':main()
