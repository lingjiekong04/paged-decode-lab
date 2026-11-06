"""Heatmap of measured medians with explicit convergence markers."""
import os,json
from _common import ROOT

def main():
    cache=ROOT/'.cache/matplotlib';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    out=ROOT/'docs/benchmark-v2'
    rows=json.loads((out/'summary.json').read_text())['rows']
    groups=[('small',256),('large',256),('standard',64),('standard',128)]
    shapes=[(b,n) for b in (1,8,32) for n in (512,2048,8192)]
    values=np.zeros((4,9));flags=np.zeros((4,9),dtype=bool)
    for i,(g,d) in enumerate(groups):
        for j,(b,n) in enumerate(shapes):
            v={r['backend']:r for r in rows if (r['geometry'],r['dim'],r['batch'],r['context'])==(g,d,b,n)}
            values[i,j]=v['flash_attn']['primary_us']/v['triton_matched']['primary_us']
            flags[i,j]=not(v['flash_attn']['converged_5pct'] and v['triton_matched']['converged_5pct'])
    fig,ax=plt.subplots(figsize=(11,3.8),layout='constrained')
    im=ax.imshow(values,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=.5,vcenter=1.,vmax=2.))
    ax.set(xticks=range(9),xticklabels=[f'B{b}\nL{n}' for b,n in shapes],yticks=range(4),
        yticklabels=['0.8B / D256','9B / D256','H32/KV8 / D64','H32/KV8 / D128'],
        title='FlashAttention / matched latency: fixed Graph128 medians')
    for i in range(4):
        for j in range(9):ax.text(j,i,f'{values[i,j]:.2f}x'+('*' if flags[i,j] else ''),ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='FA / matched (>1 favors matched)')
    fig.supxlabel('* >5% Graph32-to128 change. Color clipped to [0.5, 2]; inspect raw ranges.',fontsize=9)
    fig.savefig(out/'matrix.png',dpi=170);fig.savefig(out/'matrix.svg')

if __name__=='__main__':main()
