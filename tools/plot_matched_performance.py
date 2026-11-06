"""Standalone scientific plot of measured operator and full-model performance."""
import json
from _common import ROOT


def main():
    import os
    cache=ROOT/'.cache/matplotlib'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder=ROOT/'docs/matched-performance'
    data=json.loads((folder/'summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    backends=('flash_attn','triton_tc_splitk','matched_splitkv')
    labels=('FlashAttention','Fixed 1024','Capacity + LSE')
    colors=('#486A9B','#C58B41','#468C72')
    for backend,label,color in zip(backends,labels,colors):
        rows=sorted([r for r in data['operator_summary'] if r['model']=='large' and r['batch']==1 and r['backend']==backend],key=lambda r:r['context'])
        axes[0].plot([str(r['context']) for r in rows],[r['graph_us'] for r in rows],marker='o',label=label,color=color)
    axes[0].set(title='Qwen3.5-9B geometry, batch 1',xlabel='KV tokens',ylabel='Operator Graph interval (us)')
    axes[0].legend(frameon=False)
    groups=[('small',1),('large',1),('small',8),('large',8)]
    x=np.arange(4)
    for i,(backend,label,color) in enumerate(zip(backends,labels,colors)):
        rows=[next(r for r in data['model_summary'] if (r['model'],r['batch'],r['backend'],r['mode'])==(m,b,backend,'graph')) for m,b in groups]
        values=[r['throughput'] for r in rows]
        errors=[[r['throughput']-r['throughput_range'][0] for r in rows],
                [r['throughput_range'][1]-r['throughput'] for r in rows]]
        axes[1].bar(x+(i-1)*.24,values,yerr=errors,capsize=2,width=.24,color=color,label=label)
    axes[1].set(xticks=x,xticklabels=['0.8B / B1','9B / B1','0.8B / B8','9B / B8'],title='Full generation with decode Graph',ylabel='Output tokens / second')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(folder/'performance.png',dpi=170)
    fig.savefig(folder/'performance.svg')


if __name__=='__main__':main()
