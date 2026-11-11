"""Recompute and plot the Block/Orb clocks from portable cached trajectories.

This reproduces the numerical clock panel, not the complete typeset Figure 5.
Usage: python paper/figures/image_clocks.py --out outputs/image_clocks
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from genrule.io.tables import read_image_clock_data
from genrule.analysis.timescales import persistence_onset


def reproduce(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation

    rows, curves = read_image_clock_data(data)
    computed=[]
    for row in rows:
        steps,values=curves[(row['run'],row['metric'])]
        result=persistence_onset(steps,values,float(row['threshold']))
        expected=float(row['onset'])
        if np.isfinite(expected):
            if result.onset!=expected: raise ValueError(f'Onset mismatch: {row["run"]} {row["metric"]}')
        elif result.status!='right_censored':
            raise ValueError(f'Censoring mismatch: {row["run"]} {row["metric"]}')
        computed.append({**row,'onset':result.onset,'status':result.status,'budget':result.budget})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'image_onsets_recomputed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(computed[0]));writer.writeheader();writer.writerows(computed)

    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':9})
    fig,ax=plt.subplots(figsize=(6.0,4.3),layout='constrained')
    colors={'rule':'#2166ac','mem':'#e6550d'}
    markers={'raw':'D','block':'s','orb':'o'}
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlim(.65,2400);ax.set_ylim(450,2e6)
    ax.axvspan(384,2400,color='#b2182b',alpha=.07,lw=0)
    # Small offsets in display points keep repeated settings visible without
    # changing the numerical d coordinates in the source table.
    repeated={}
    for row in computed:
        key=(row['d'],row['metric'])
        repeated.setdefault(key,[]).append(row)
    for row in computed:
        key=(row['d'],row['metric']);peers=repeated[key]
        i=peers.index(row)
        offset=(-2 if row['metric']=='rule' else 2)+(i-(len(peers)-1)/2)*5
        transform=ax.transData+ScaledTranslation(offset/72,0,fig.dpi_scale_trans)
        x=float(row['d']);color=colors[row['metric']]
        if row['status']=='observed':
            ax.plot(x,row['onset'],marker=markers[row['dataset']],ls='none',ms=5,
                    color=color,mec='white',mew=.4,transform=transform)
        else:
            y=row['budget']
            ax.plot([x*.93,x*1.07],[y,y],color=color,lw=.65,ls='--',transform=transform)
            ax.annotate('',xy=(x,y*1.3),xytext=(x,y),xycoords=transform,
                        arrowprops={'arrowstyle':'->','color':color,'lw':.7})
    fit=[r for r in computed if r['dataset'] in ('raw','block') and r['metric']=='rule' and float(r['d'])<=192 and r['onset'] is not None]
    x=np.array([float(r['d']) for r in fit]);y=np.array([r['onset'] for r in fit])
    slope,intercept=np.polyfit(np.log10(x),np.log10(y),1)
    grid=np.geomspace(x.min(),x.max(),100)
    ax.plot(grid,10**intercept*grid**slope,color=colors['rule'],lw=.8)
    (out/'fit.json').write_text(json.dumps({'slope':float(slope),'log10_intercept':float(intercept),'n':len(fit),'d_max':192},indent=2)+'\n')
    ax.set(xlabel='Input dimension per token d',ylabel='Training step',
           title='Block / Orb clocks: parity G=2, N=2048, D=T=36')
    legend=[Line2D([],[],color=colors[m],label=f'tau_{m}') for m in ('rule','mem')]
    legend += [Line2D([],[],color='0.3',marker=markers[d],ls='none',label=d.title()) for d in ('raw','block','orb')]
    ax.legend(handles=legend,fontsize=8,loc='upper left',ncol=2)
    ax.text(.98,.02,'Dashed cap + arrow: no onset within budget\nSmall horizontal offsets separate overlapping settings',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=7)
    for ext in ('pdf','png','svg'):fig.savefig(out/f'image_clocks.{ext}',dpi=180)
    plt.close(fig)
    print(f'Reproduced {len(computed)} onsets/bounds; slope={slope:.8f}; outputs: {out}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--out',type=Path,default=Path('outputs/image_clocks'))
    args=parser.parse_args()
    reproduce(args.data,args.out)
