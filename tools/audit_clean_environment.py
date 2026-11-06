"""Check actual module origins and shared-object paths in a fresh venv."""
import argparse,importlib,importlib.metadata,json,os,sys,sysconfig
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--forbid',action='append',default=[])
    p.add_argument('--gpu',action='store_true')
    args=p.parse_args()
    prefix=Path(sys.prefix).resolve()
    if prefix==Path(sys.base_prefix).resolve():raise RuntimeError('a virtual environment is required')
    forbidden=[Path(x).resolve() for x in args.forbid]
    paths=[Path(x).resolve() for x in sys.path if x]
    assert not any(x.is_relative_to(bad) for x in paths for bad in forbidden),'forbidden import search path'
    modules={}
    for name in ('torch','triton','flash_attn','flash_attn_2_cuda','transformers','tokenizers','numpy','einops','pytest'):
        module=importlib.import_module(name);origin=Path(module.__file__).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin),'outside fresh venv')
        modules[name]=str(origin)
    project=importlib.import_module('paged_decode')
    project_origin=Path(project.__file__).resolve()
    assert project_origin.is_relative_to(Path(__file__).resolve().parents[1]),'project imported from another checkout'
    modules['paged_decode']=str(project_origin)
    pth={str(f):f.read_text() for f in Path(sysconfig.get_path('purelib')).glob('*.pth')}
    assert not any(str(bad) in text for text in pth.values() for bad in forbidden),'forbidden .pth reference'
    import torch
    assert torch.__version__=='2.6.0+cu124' and torch.version.cuda=='12.4'
    assert not torch._C._GLIBCXX_USE_CXX11_ABI,'FlashAttention wheel requires CXX11 ABI false'
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    result={'prefix':str(prefix),'base_prefix':sys.base_prefix,'sys_path':list(map(str,paths)),
        'modules':modules,'packages':packages,'forbidden_paths':list(map(str,forbidden)),
        'pythonpath':os.environ.get('PYTHONPATH'),'python_no_user_site':os.environ.get('PYTHONNOUSERSITE'),
        'pth_files':pth,
        'torch_cuda':torch.version.cuda,'cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI}
    if args.gpu:
        assert torch.cuda.is_available()
        x=torch.ones(1,device='cuda');torch.cuda.synchronize()
        result['gpu']={'name':torch.cuda.get_device_name(),'probe':x.item()}
    maps=Path('/proc/self/maps')
    if maps.exists():
        loaded=sorted({line.split()[-1] for line in maps.read_text().splitlines() if line.split()[-1].startswith('/')})
        assert not any(Path(x).resolve().is_relative_to(bad) for x in loaded for bad in forbidden),'library mapped from forbidden environment'
        result['mapped_library_paths']=[x for x in loaded if '.so' in x]
    result['status']='passed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    args.output.with_suffix('.requirements.txt').write_bytes(('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items()))+'\n').encode())
    print('Fresh venv module origins, Torch CUDA and ABI checks passed')


if __name__=='__main__':main()
