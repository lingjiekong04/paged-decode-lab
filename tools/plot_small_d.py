"""Plot measured complete and isolated-stage intervals separately, without stacking."""
import json,os
from pathlib import Path
from _common import ROOT


def main():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder=ROOT/'docs/small-d';data=json.loads((folder/'summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    variants=('native','matched_api','partition_only','merge_only')
    colors=('#385D8A','#4D917C','#D48C48','#9D78AC')
    labels=('Native','Matched complete','Partition alone','Merge alone*')
    cases=((1,512),(1,8192),(32,8192));x=np.arange(3)
    for ax,d in zip(axes,(64,128,256)):
        for i,(v,label,color) in enumerate(zip(variants,labels,colors)):
            rows=[next(r for r in data['graph_rows'] if (r['dim'],r['batch'],r['context'],r['variant'])==(d,b,n,v)) for b,n in cases]
            y=np.array([r['primary_us'] for r in rows]);lo=np.array([r['range_us'][0] for r in rows]);hi=np.array([r['range_us'][1] for r in rows])
            ax.bar(x+(i-1.5)*.19,y,.18,label=label,color=color,yerr=np.array([y-lo,hi-y]),capsize=2)
        ax.set(yscale='log',xticks=x,xticklabels=('B1 / L512','B1 / L8192','B32 / L8192'),title=f'D={d}, Hq=32 / Hkv=8',ylabel='Graph interval per operator (us, log scale)')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Unroll128; five randomized rounds; isolated stages are not an additive decomposition\n*Merge: 32/128 delta > 5%, absolute interval not converged',fontsize=12)
    fig.savefig(folder/'stages.png',dpi=180);fig.savefig(folder/'stages.svg');plt.close(fig)


if __name__=='__main__':main()
