"""Run the public CLI twice and check Graph generation; this is not a benchmark."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from _common import ROOT,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=32)
    args=p.parse_args()
    if args.tokens<=0:p.error('--tokens must be positive')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONNOUSERSITE':'1'}
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    data={}
    for backend in ('flash_attn','triton_matched'):
        destination=args.output/backend
        cmd=[sys.executable,str(ROOT/'tools/qwen35.py'),'--model',str(args.model.resolve()),
             '--backend',backend,'--mode','graph','--tokens',str(args.tokens),'--output',str(destination)]
        with (args.output/f'{backend}.log').open('x') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        read=lambda name:json.loads((destination/name).read_text())
        result=read('summary.json');runtime=read('runtime.json')
        assert result['status']=='completed' and runtime['backend']==backend and runtime['mode']=='graph'
        data[backend]={'result':result['result'],'runtime':runtime,'model':read('arguments.json')['model']}
    native,matched=data['flash_attn'],data['triton_matched']
    assert native['model']==matched['model'],'model identity differs'
    equal=native['result']['token_ids']==matched['result']['token_ids']
    report={'status':'passed' if equal else 'failed','purpose':'Graph greedy demonstration, not performance measurement',
        'model':native['model'],'mode':'graph','batch':1,'requested_tokens':args.tokens,
        'generated_tokens':sum(map(len,matched['result']['token_ids'])),'token_ids_equal':equal,
        'text':matched['result']['text'],'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'configs':{k:v['runtime'] for k,v in data.items()}}
    write_json(args.output/'comparison.json',report)
    print(json.dumps({k:report[k] for k in ('status','generated_tokens','token_ids_equal','text')},ensure_ascii=False))
    if not equal:raise SystemExit(2)


if __name__=='__main__':main()
