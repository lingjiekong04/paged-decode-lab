"""Plot graph-unroll convergence; no model speedup implied."""
import os,json
from _common import ROOT

def main():
    cache=ROOT/'.cache/matplotlib';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'docs/short-overhead'
    rows=json.loads((out/'summary.json').read_text())['graph_rows']
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained',sharey=True)
    for ax,model,title in zip(axes,('small','large'),('0.8B geometry','9B geometry')):
        for variant,label in [('native','FlashAttention'),('matched_api','Matched Triton'),('merge_only','Merge only (diagnostic)')]:
            selected=sorted([r for r in rows if (r['model'],r['context'],r['variant'])==(model,512,variant)],key=lambda r:r['unroll'])
            ax.plot([str(r['unroll']) for r in selected],[r['interval_us'] for r in selected],marker='o',label=label)
        ax.set(title=title+', batch 1 / KV 512',xlabel='Operators inside one Graph')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
    axes[0].set_ylabel('CUDA-event interval per operator (us)')
    axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(out/'convergence.png',dpi=170);fig.savefig(out/'convergence.svg')

if __name__=='__main__':main()
