"""Rehash an existing pinned model snapshot without importing a model runtime."""
import argparse,hashlib,json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    model=args.model.resolve()
    manifest=json.loads((model/'paged_decode_model.json').read_text())
    specs=json.loads((Path(__file__).resolve().parents[1]/'configs/models.json').read_text())
    spec=next(s for s in specs.values() if s['repo_id']==manifest['repo_id'])
    assert manifest['revision']==spec['revision'] and manifest['weights']==spec['weights']
    checked=[]
    for entry in spec['weights']:
        path=(model/entry['name']).resolve()
        assert path.is_relative_to(model)
        assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],str(path)
        checked.append(entry)
    for name,sha in manifest['files'].items():
        path=(model/name).resolve()
        assert path.is_relative_to(model) and digest(path)==sha,str(path)
    result={'status':'passed','model_directory':str(model),'repo_id':spec['repo_id'],
        'revision':spec['revision'],'weights':checked,'files':manifest['files']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Pinned model snapshot SHA256 verified:',spec['repo_id'])


if __name__=='__main__':main()
