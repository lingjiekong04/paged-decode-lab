"""Check release files and reject unlisted non-local extras; manifest excludes itself."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRS = {'.git', '.cache', '.pytest_cache', '__pycache__', 'models', 'downloads', 'results', 'build', 'dist'}

def local_path(rel):
    return any(p in LOCAL_DIRS or p.startswith('.venv') or p.endswith('.egg-info') for p in rel.parts) or rel.parts[:2] == ('vendor', 'nano-vllm') or rel.suffix in {'.pyc', '.pyo'}

def verify():
    m = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    expected = {r['path']: r for r in m['files']}
    assert len(expected) == len(m['files']), 'duplicate manifest entries'
    for rel, r in expected.items():
        p = Path(rel)
        assert not p.is_absolute() and '..' not in p.parts
        data = (ROOT / p).read_bytes()
        assert len(data) == r['bytes'], rel
        assert hashlib.sha256(data).hexdigest() == r['sha256'], rel
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and not local_path(p.relative_to(ROOT))}
    assert actual == set(expected) | {'RELEASE_MANIFEST.json'}, ('unlisted/missing files', actual ^ (set(expected) | {'RELEASE_MANIFEST.json'}))
    print(f'Release manifest passed: {len(expected)} indexed files, self explicitly excluded.')

if __name__ == '__main__':
    verify()
