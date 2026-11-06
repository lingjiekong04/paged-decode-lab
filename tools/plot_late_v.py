"""Standalone V-load placement plot, displaying five-round min/max."""
import json,os
from _common import ROOT


def main():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder=ROOT/'docs/late-v';data=json.loads((folder/'summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(11.8,4.2),layout='constrained')
    cases=((1,512),(1,8192),(32,8192));x=np.arange(3)
    for ax,d in zip(axes,(64,128,256)):
        for i,(w,color) in enumerate((('baseline','#3F6C97'),('late_v','#BD7747'))):
            rows=[next(r for r in data['rows'] if (r['dim'],r['batch'],r['context'],r['stage'])==(d,b,n,'complete')) for b,n in cases]
            values=[r['variants'][w] for r in rows];y=np.array([r['median_us'] for r in values])
            lo=np.array([r['range_us'][0] for r in values]);hi=np.array([r['range_us'][1] for r in values])
            ax.bar(x+(i-.5)*.32,y,.3,color=color,label='Original V-load placement' if w=='baseline' else 'Source V load moved before PV',yerr=np.array([y-lo,hi-y]),capsize=3)
            for j,r in enumerate(values):
                if not r['converged_5pct']:ax.text(x[j]+(i-.5)*.32,hi[j]*1.05,'*',ha='center')
        ax.set(title=f'D={d}, Hq=32 / Hkv=8',yscale='log',ylabel='Complete Graph interval (us, log scale)',
            xticks=x,xticklabels=('B1 / L512','B1 / L8192','B32 / L8192'))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Identical execution code in 9/9 cases; only source V-load placement changes.\nUnroll128; five randomized rounds; error bars min/max; * marks convergence failure.',fontsize=12)
    fig.savefig(folder/'latency.png',dpi=180);fig.savefig(folder/'latency.svg');plt.close(fig)


if __name__=='__main__':main()

