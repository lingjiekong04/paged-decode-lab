"""Targeted audit of retained Triton snapshots; lexical spans are not register liveness."""
import hashlib,json,re
from collections import Counter
from pathlib import Path
from _common import ROOT

SSA=r'%[A-Za-z0-9_]+(?:#\d+)?'


def strip_locations(line):
    while 'loc(' in line:
        start=line.index('loc(');i=start+4;depth=1;quoted=False;escape=False
        while i<len(line) and depth:
            c=line[i]
            if escape:escape=False
            elif c=='\\' and quoted:escape=True
            elif c=='"':quoted=not quoted
            elif not quoted:depth+=(c=='(')-(c==')')
            i+=1
        if depth:raise ValueError('unbalanced location')
        line=line[:start]+line[i:]
    return line.rstrip()


def normalize(text):
    lines=[strip_locations(s) for s in text.splitlines() if not re.match(r'^#loc\w*\s*=',s)]
    body='\n'.join(lines).replace('@_partition_late_v','@partition').replace('@_normalized_partition','@partition')
    names={}
    def rename(m):
        name=m[0]
        if name not in names:names[name]=f'%v{len(names)}'
        return names[name]
    body=re.sub(r'%[A-Za-z0-9_]+',rename,body)
    return '\n'.join(re.sub(r'\s+',' ',s).strip() for s in body.splitlines() if s.strip())+'\n'


def loop_bounds(lines):
    starts=[i for i,s in enumerate(lines) if ' = scf.for ' in s]
    if len(starts)!=1:raise ValueError('expected one tile loop')
    start=starts[0];indent=re.match(r'\s*',lines[start])[0]
    end=next(i for i in range(start+1,len(lines)) if lines[i].startswith(indent+'}'))
    return start,end,len(indent)+2


def op_records(lines,start,end,indent):
    result=[]
    for i in range(start+1,end):
        line=strip_locations(lines[i]);space=len(line)-len(line.lstrip())
        if space!=indent:continue  # Nested reduce regions reuse local SSA names.
        m=re.match(r'\s*('+SSA+r'(?:,\s*'+SSA+r')*)\s*=\s*(.*)',line)
        if m:result.append({'line':i+1,'defs':re.findall(SSA,m[1]),'rhs':m[2],'text':line.strip()})
        elif line.strip().startswith('scf.yield'):result.append({'line':i+1,'defs':[],'rhs':line.strip(),'text':line.strip()})
    return result


def span(value,definition,ops):
    uses=[o['line'] for o in ops if value in re.findall(SSA,o['rhs'])]
    return {'value':value,'definition_line':definition,'use_lines':uses,'last_use_line':max(uses) if uses else definition,
        'scope':'Lexical def/use in one printed loop body; not cycles, physical registers or a CFG liveness solution'}


def inspect(path):
    lines=path.read_text(encoding='utf-8').splitlines();start,end,indent=loop_bounds(lines)
    ops=op_records(lines,start,end,indent);defs={v:o for o in ops for v in o['defs']}
    dots=[o for o in ops if o['rhs'].startswith('tt.dot ')]
    assert len(dots)==2
    qk,pv=dots;pv_inputs=re.findall(SSA,pv['rhs'].split('inputPrecision')[0])
    qk_inputs=re.findall(SSA,qk['rhs'].split('inputPrecision')[0])
    probabilities=[o for o in ops if o['rhs'].startswith('math.exp2 ') and re.search(r'tensor<16x\d+xf32',o['rhs'])]
    assert len(probabilities)==1
    prob=probabilities[0];p=prob['defs'][0]
    trunc=next(o for o in ops if o['rhs'].startswith('arith.truncf '+p+' '))
    denom=[o for o in ops if prob['line']<o['line']<trunc['line']]
    conversions=[o for o in denom if 'triton_gpu.convert_layout' in o['rhs'] and '-> tensor<16x4xf32, #mma>' in o['rhs']]
    reduces=[o for o in denom if o['rhs'].startswith('"tt.reduce"')]
    masks=[]
    for o in denom:
        if o['rhs'].startswith('arith.select ') and ('%outLHS' in o['rhs'] or '%outRHS' in o['rhs']):
            masks.append(re.findall(SSA,o['rhs'])[0])
    tracked={}
    for label,value in [('Q_fragment',qk_inputs[0]),('K_fragment',qk_inputs[1]),('V_fragment',pv_inputs[1]),('probabilities_fp32',p),('PV_operand',pv_inputs[0])]:
        if value in defs:d=defs[value]
        else:
            matches=[(i,s) for i,s in enumerate(lines[:start]) if re.match(r'\s*'+re.escape(value)+r'\s*=',s)]
            assert len(matches)==1,(path,value)
            i,s=matches[0];d={'line':i+1,'text':strip_locations(s).strip()}
        tracked[label]={**span(value,d['line'],ops),'definition':d['text']}
    scaled_acc=defs[pv_inputs[2]];carried=re.findall(SSA,scaled_acc['rhs'])[0]
    tracked['loop_carried_accumulator']={**span(carried,start+1,ops),'definition':'scf.for iter_args; state persists between iterations'}
    allocs=[]
    for i,line in enumerate(lines):
        if 'triton_gpu.local_alloc' not in line:continue
        m=re.search(r'!tt.memdesc<([0-9x]+)x(bf16|i32)',line)
        assert m,line
        shape=list(map(int,m[1].split('x')));elements=1
        for n in shape:elements*=n
        allocs.append({'line':i+1,'shape':shape,'dtype':m[2],'logical_bytes':elements*(2 if m[2]=='bf16' else 4),
            'inside_loop':start<i<end,'text':strip_locations(line).strip()})
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'loop_start_line':start+1,'loop_end_line':end+1,'qk':qk,'pv':pv,'probability':prob,'bf16_conversion':trunc,
        'denominator_reductions':len(reduces),'denominator_to_mma_conversions':len(conversions),
        'mask_values':sorted(set(masks)),'mask_count':len(set(masks)),'tracked':tracked,'shared_allocations':allocs,
        'async_copies':[{'line':i+1,'text':strip_locations(s).strip(),'inside_loop':start<i<end} for i,s in enumerate(lines) if 'async_copy_global_to_local' in s],
        'async_waits':[{'line':i+1,'text':strip_locations(s).strip()} for i,s in enumerate(lines) if 'triton_gpu.async_wait' in s],
        'layouts':[s for s in lines if re.match(r'^#(?:mma|blocked\d*|shared\d*) =',s)]}


def llvm_origin(metadata, identifier):
    """Follow inlinedAt to the outer source location, then resolve its scope file."""
    seen=set()
    while True:
        if identifier in seen:raise ValueError('cyclic debug location')
        seen.add(identifier);value=metadata.get(identifier,'')
        inline=re.search(r'inlinedAt: !(\d+)',value)
        if not inline:break
        identifier=inline[1]
    line=re.search(r'!DILocation\(line: (\d+)',value)
    scope=re.search(r'scope: !(\d+)',value)
    filename=None;seen=set()
    while scope:
        identifier=scope[1]
        if identifier in seen:raise ValueError('cyclic debug scope')
        seen.add(identifier);value=metadata.get(identifier,'')
        file_match=re.search(r'!DIFile\(filename: "([^"]+)"',value)
        if file_match:filename=file_match[1];break
        scope=re.search(r'file: !(\d+)',value) or re.search(r'scope: !(\d+)',value)
    return {'file':filename,'line':int(line[1]) if line else None}


def lower_level_sites(path):
    """Count printed instruction/call sites, never dynamic execution counts."""
    lines=path.read_text(encoding='utf-8').splitlines()
    metadata=dict(re.findall(r'^!(\d+) = (.*)$','\n'.join(lines),re.M))
    groups={'shuffle':[], 'mma_bf16':[], 'ldmatrix':[], 'async_copy':[],
            'async_commit':[], 'async_wait':[]}
    ptx_location={};ptx_files={}
    if path.suffix=='.ptx':
        ptx_files=dict(re.findall(r'^\s*\.file\s+(\d+)\s+"([^"]+)"','\n'.join(lines),re.M))
    for i,text in enumerate(lines):
        if path.suffix=='.llir' and not re.search(r'\bcall\b',text):continue
        if path.suffix=='.ptx':
            loc=re.match(r'\s*\.loc\s+(\d+)\s+(\d+)',text)
            if loc:ptx_location={'file':Path(ptx_files.get(loc[1],'unknown')).name,'line':int(loc[2])}
            if text.lstrip().startswith(('//','.')):continue
        kind=None
        if 'shfl.sync.' in text:kind='shuffle'
        elif 'mma.sync.' in text and '.bf16.bf16.' in text:kind='mma_bf16'
        elif 'ldmatrix.sync.' in text:kind='ldmatrix'
        elif 'cp.async.commit_group' in text:kind='async_commit'
        elif 'cp.async.wait_group' in text:kind='async_wait'
        elif re.search(r'cp\.async\.c[ag]\.shared\.global',text):kind='async_copy'
        if kind:
            dbg=re.search(r'!dbg !(\d+)',text)
            origin=llvm_origin(metadata,dbg[1]) if dbg else (dict(ptx_location) if path.suffix=='.ptx' else {})
            groups[kind].append({'line':i+1,'origin':origin,'text':text.strip()})
    totals={k:len(v) for k,v in groups.items()}
    by_source={k:dict(sorted(Counter(str(x['origin'].get('file'))+':'+str(x['origin'].get('line')) for x in v).items())) for k,v in groups.items()}
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'counts':totals,'by_source':by_source,
            'examples':{k:(v[:2]+v[-2:] if len(v)>4 else v) for k,v in groups.items()},
            'scope':'Static printed sites; LLVM/PTX, not SASS, cycles, executed instructions or hardware counters'}


def main():
    source=ROOT/'results/latev-ir';manifest=json.loads((source/'manifest.json').read_text())
    verified=[]
    for entry in manifest['entries']:
        for name,info in entry['files'].items():
            data=(source/name).read_bytes()
            assert hashlib.sha256(data).hexdigest()==info['sha256'] and len(data)==info['bytes'],name
            verified.append(name)
    out=ROOT/'docs/compiler-ir';out.mkdir(parents=True,exist_ok=True)
    analyses=[];comparisons=[];lower=[]
    for case in ('b1-l8192-d64','b1-l8192-d128'):
        for stage in ('ttir','ttgir'):
            normalized=[]
            for variant in ('baseline','late_v'):
                path=source/case/(variant+'.'+stage);analysis=inspect(path);analyses.append(analysis)
                text=normalize(path.read_text(encoding='utf-8'));normalized.append(text)
                target=out/'normalized'/case/(variant+'.'+stage);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(text.encode())
            comparisons.append({'case':case,'stage':stage,'normalized_text_equal':normalized[0]==normalized[1],
                'note':'Debug locations and printed SSA names removed; this is a text comparison, not an IR equivalence proof'})
        for variant in ('baseline','late_v'):
            for stage in ('llir','ptx'):lower.append(lower_level_sites(source/case/(variant+'.'+stage)))
    result={'analyses':analyses,'comparisons':comparisons,'lower_level':lower,'kernel_changed':False,
        'verified_raw_files':sorted(verified),
        'manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        'scope':'Offline analysis of retained compiler snapshots; no new GPU timing or hardware counters'}
    (out/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(comparisons))


if __name__=='__main__':main()
