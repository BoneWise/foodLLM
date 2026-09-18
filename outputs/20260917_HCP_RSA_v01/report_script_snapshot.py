"""Plot all retained HCP parcels and paired regional contrasts; descriptive only."""
import csv, json, shutil, platform, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_roi import sha

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'outputs/20260917_HCP_RSA_v01';a=json.loads((p/'audit.json').read_text());assert a['status']=='complete'
    ix=json.loads((p/'index.json').read_text());names=ix['series'];nu=ix['nutrients'];layers=np.arange(1,33)
    for filename in ['bilateral_and_reference_and_group_scores.npy','bilateral_and_reference_and_group_contrasts.npy','regional_paired_contrasts.npy']:
        assert sha(p/filename)==a['output_sha256'][filename]
    sc=np.load(p/'bilateral_and_reference_and_group_scores.npy');co=np.load(p/'bilateral_and_reference_and_group_contrasts.npy');reg=np.load(p/'regional_paired_contrasts.npy')
    out=p/'figures';out.mkdir(exist_ok=False)
    labels=['Protein','Fat','Carbohydrate','Fibre'];colors=['#6d3da8','#007f83','#bf731b']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def finish(fig,name):
        fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    def basics(ax):ax.axhline(0,color='#888888',lw=.7);ax.set_xlim(1,32);ax.set_xticks([1,8,16,24,32]);ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    region=['OFC_nine','higher_visual','early_intermediate_visual']
    fig,axs=plt.subplots(4,2,figsize=(12,13),sharex=True)
    for ni,nut in enumerate(nu):
        for ci,c in enumerate(['T - V','T - mean(P1..P5)']):
            ax=axs[ni,ci]
            for ri,r in enumerate(region):
                for pi,pr in enumerate(['A','B']):ax.plot(layers,co[pi,ni,names.index(r),:,:,ci].mean(0),color=colors[ri],ls='-' if pi==0 else '--',label=f'{r} / {pr}')
            basics(ax);ax.set_title(labels[ni]+' | '+c);ax.set_ylabel('Change in Spearman rho')
    axs[0,0].legend(fontsize=8,ncol=2);axs[-1,0].set_xlabel('Decoder layer');axs[-1,1].set_xlabel('Decoder layer')
    fig.suptitle('Regional alignment gains | 29 paired participants | A primary; B robustness\nEqual hemisphere scores, then equal parcels; lines are participant means',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));finish(fig,'regional_gains')
    # Main region-to-region difference with full participant distributions on the pre-specified 32-layer mean.
    fig,axs=plt.subplots(4,3,figsize=(15,12))
    for ni in range(4):
        for ci in range(2):
            ax=axs[ni,ci]
            for pi,pr in enumerate(['A','B']):
                y=reg[pi,ni,0,:,:,ci];ax.plot(layers,y.mean(0),color=['#6d3da8','#007f83'][pi],ls=['-','--'][pi],label='Prompt '+pr)
            basics(ax);ax.set_title(labels[ni]+' | '+['T - V','T - meanP'][ci]);ax.set_ylabel('OFC gain - high visual gain')
        ax=axs[ni,2]
        for k,(pi,ci) in enumerate([(0,0),(1,0),(0,1),(1,1)]):
            v=reg[pi,ni,0,:,:,ci].mean(1);offset=np.linspace(-.12,.12,len(v))
            ax.scatter(k+offset,v,s=15,alpha=.65,color=['#6d3da8','#007f83'][pi]);ax.plot([k-.2,k+.2],[v.mean()]*2,color='black',lw=2)
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(4),['A T-V','B T-V','A T-P','B T-P']);ax.set_title('32-layer mean / each participant');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    axs[0,0].legend();fig.suptitle('Primary paired regional comparison | descriptive, no significance tests\nCurves: group means. Dots: participants. Black bars: arithmetic means.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));finish(fig,'OFC_minus_higher_visual')
    ofc=ix['groups']['OFC_nine']
    for ni,nut in enumerate(nu):
        fig,axs=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True)
        for ax,r in zip(axs.flat,ofc):
            for ci,c in enumerate(['T-V','T-meanP']):
                for pi,pr in enumerate(['A','B']):ax.plot(layers,co[pi,ni,names.index(r),:,:,ci].mean(0),color=['#6d3da8','#007f83'][ci],ls=['-','--'][pi],label=pr+' '+c)
            basics(ax);ax.set_title(r)
        axs[0,0].legend(fontsize=8,ncol=2);fig.suptitle(labels[ni]+' | all nine OFC parcels retained | 29 participant means\nSame vertical scale across all nine panels; solid A, dashed B',fontsize=13)
        fig.supxlabel('Decoder layer');fig.supylabel('Change in Spearman rho');fig.tight_layout(rect=(.02,.02,1,.94));finish(fig,nut+'_nine_OFC_parcels')
    # All 18 parcels and the 3 old ROI references in a common-scale heatmap.
    for pi,pr in enumerate(['A','B']):
        values=co[pi,:,:21].mean(axis=2) # nutrient, ROI, layer, contrast
        bound=float(np.abs(values).max())
        fig,axs=plt.subplots(2,4,figsize=(16,11),sharex=True,sharey=True)
        for ni in range(4):
            for ci in range(2):
                ax=axs[ci,ni];im=ax.imshow(values[ni,:,:,ci],aspect='auto',origin='upper',cmap='RdBu_r',vmin=-bound,vmax=bound,extent=[.5,32.5,20.5,-.5])
                ax.axhline(17.5,color='black',lw=1);ax.set_yticks(range(21),names[:21]);ax.set_xticks([1,8,16,24,32]);ax.set_title(labels[ni]+' | '+['T-V','T-meanP'][ci]);ax.set_xlabel('Layer')
        fig.suptitle(f'Prompt {pr} | all 18 HCP bilateral parcels + 3 unchanged legacy references\nCommon color scale; boundary separates HCP and legacy ROI definitions',fontsize=13)
        fig.tight_layout(rect=(0,0,.94,.94));cb=fig.add_axes([.955,.2,.012,.55]);fig.colorbar(im,cax=cb,label='Mean change in Spearman rho');finish(fig,'all_ROI_heatmap_'+pr)
    # Absolute scores accompany gains; P envelope is five assignment means, not uncertainty.
    for pi,pr in enumerate(['A','B']):
        fig,axs=plt.subplots(3,4,figsize=(16,10),sharex=True)
        for ri,r in enumerate(region):
            for ni in range(4):
                ax=axs[ri,ni];avg=sc[pi,ni,names.index(r)].mean(0)
                for ci,c in enumerate(['V','T','meanP']):ax.plot(layers,avg[:,ci] if ci<2 else avg[:,2:].mean(1),color=['#6c757d','#6d3da8','#007f83'][ci],label=c)
                ax.fill_between(layers,avg[:,2:].min(1),avg[:,2:].max(1),color='#007f83',alpha=.12)
                basics(ax);ax.set_title(labels[ni]+' | '+r);ax.set_ylabel('Spearman rho')
        axs[0,0].legend();fig.suptitle(f'Prompt {pr} | absolute model-brain RSA\nShading is P1-P5 range of group means, not a confidence interval',fontsize=13);fig.tight_layout(rect=(0,0,1,.94));finish(fig,'absolute_regional_alignment_'+pr)
    qc=list(csv.DictReader(open(p/'acquisition_and_consistency_QC.csv')))
    metrics=['session_RDM_rho','run_RDM_rho','LOSO_group_RDM_rho_not_test_retest']
    fig,axs=plt.subplots(1,3,figsize=(14,9),sharey=True)
    for mi,metric in enumerate(metrics):
        vals=[]
        for r in names[:21]:
            selected=[r] if r in ['vmPFC','lOFC','mOFC'] else ['L_'+r,'R_'+r]
            vals.append(np.mean([float(x['value']) for x in qc if x['ROI'] in selected and x['metric']==metric]))
        ax=axs[mi];ax.barh(np.arange(21),vals,color='#587696');ax.invert_yaxis();ax.set_yticks(range(21),names[:21]);ax.axvline(0,color='gray',lw=.7);ax.axhline(17.5,color='black',lw=.7)
        ax.set_title(['Session structure','Session + run structure','LOSO between-person consistency'][mi]);ax.set_xlabel('Mean Spearman rho')
    fig.suptitle('Quality context | HCP hemisphere means; legacy ROIs unchanged\nLOSO consistency is not test-retest reliability or a normalization factor',fontsize=13);fig.tight_layout(rect=(0,0,1,.93));finish(fig,'ROI_quality_context')
    shutil.copy2(__file__,p/'report_script_snapshot.py')
    audit=dict(status='plots_complete_visual_review_pending',created=datetime.now().astimezone().isoformat(),input_RSA_audit_sha256=sha(p/'audit.json'),script_sha256=sha(__file__),seed=None,
      descriptive_only=True,git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),versions=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
      output_sha256={str(f.relative_to(p)):sha(f) for f in out.iterdir()})
    (p/'report_audit.json').write_text(json.dumps(audit,indent=2));print('HCP figures complete')

if __name__=='__main__':main()
