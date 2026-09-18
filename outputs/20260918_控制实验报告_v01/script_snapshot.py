"""Descriptive references, frozen matching and plots; no significance inference."""
import csv,json,shutil,platform
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_roi import sha,write_csv
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def main():
 cfg=read(ROOT/'configs/controls_v09_execution_v01.json');p=ROOT/cfg['RSA_output'];a=read(p/'audit.json');ix=read(p/'index.json');assert a['status']=='complete'
 out=ROOT/'outputs/20260918_控制实验报告_v01';out.mkdir(exist_ok=False);figdir=out/'figures';figdir.mkdir();shutil.copy2(__file__,out/'script_snapshot.py')
 names=ix['series'];parcels=ix['parcels'];random=ix['random_parcels'];original=ix['original_parcels'];groups=ix['groups'];prompts=ix['prompts'];nutrients=ix['nutrients'];subjects=ix['subject_ids']
 for f in ['series_VTP_scores.npy','series_N_scores.npy','series_contrasts.npy','regional_paired_contrasts.npy']:assert sha(p/f)==a['output_sha256'][f]
 co=np.load(p/'series_contrasts.npy');vtp=np.load(p/'series_VTP_scores.npy');ns=np.load(p/'series_N_scores.npy');regional=np.load(p/'regional_paired_contrasts.npy')
 qc=rows(p/'acquisition_and_consistency_QC.csv');coverage=rows(ROOT/'outputs/20260917_HCP图谱准备_v01/coverage.csv')+rows(ROOT/cfg['atlas_preparation']/'coverage.csv')
 metrics={};subject_metrics=[]
 for name in parcels:
  rec=[]
  for sub in subjects:
   cc=[r for r in coverage if r['subject_id']==sub and r['ROI'] in ['L_'+name,'R_'+name]];assert len(cc)==2
   rr=dict(ROI=name,subject_id=sub,fixed_voxels=float(np.mean([int(c['fixed_voxels']) for c in cc])),coverage=float(np.mean([float(c['coverage']) for c in cc])))
   for metric,key in [('LOSO_group_RDM_rho_not_test_retest','LOSO'),('session_RDM_rho','session'),('run_RDM_rho','run')]:
    vals=[float(r['value']) for r in qc if r['subject_id']==sub and r['ROI'] in ['L_'+name,'R_'+name] and r['metric']==metric];assert len(vals)==2;rr[key]=float(np.mean(vals))
   rec.append(rr)
  subject_metrics+=rec
  metrics[name]=dict(ROI=name,kind='random_HCP' if name in random else 'original_HCP',LOSO=float(np.mean([r['LOSO'] for r in rec])),median_fixed_voxels=float(np.median([r['fixed_voxels'] for r in rec])),mean_coverage=float(np.mean([r['coverage'] for r in rec])),session=float(np.mean([r['session'] for r in rec])),run=float(np.mean([r['run'] for r in rec])))
 for name in ['lOFC','mOFC','vmPFC']:
  rr=dict(ROI=name,kind='legacy_different_measurement_unit',median_fixed_voxels='',mean_coverage='')
  for metric,key in [('LOSO_group_RDM_rho_not_test_retest','LOSO'),('session_RDM_rho','session'),('run_RDM_rho','run')]:rr[key]=float(np.mean([float(r['value']) for r in qc if r['ROI']==name and r['metric']==metric]))
  metrics[name]=rr
 for name,pp in groups.items():
  rec=[r for r in subject_metrics if r['ROI'] in pp]
  metrics[name]=dict(ROI=name,kind='fixed_group_summary',LOSO=float(np.mean([metrics[x]['LOSO'] for x in pp])),median_fixed_voxels=float(np.median([np.mean([r['fixed_voxels'] for r in rec if r['subject_id']==sub]) for sub in subjects])),mean_coverage=float(np.mean([metrics[x]['mean_coverage'] for x in pp])),session=float(np.mean([metrics[x]['session'] for x in pp])),run=float(np.mean([metrics[x]['run'] for x in pp])))
 write_csv(out/'ROI_quality_summary.csv',list(metrics.values()));write_csv(out/'HCP_subject_quality.csv',subject_metrics)
 # Fixed calipers; low comparability is a result, never a trigger to widen them.
 rule=cfg['matching'];matched={}
 for name in original:
  target=metrics[name]
  matched[name]=[r for r in random if abs(metrics[r]['LOSO']-target['LOSO'])<=rule['LOSO_absolute_caliper'] and rule['median_fixed_voxel_ratio_range'][0]<=metrics[r]['median_fixed_voxels']/target['median_fixed_voxels']<=rule['median_fixed_voxel_ratio_range'][1]]
 (out/'matched_reference_members.json').write_text(json.dumps(matched,indent=2))
 ranks=[]
 for pi,pr in enumerate(prompts):
  for nj,nut in enumerate(nutrients):
   for window,ls in ix['windows'].items():
    vals=co[pi,nj][:,:,np.array(ls)-1,:].mean(axis=(1,2))
    for ci,c in enumerate(ix['contrasts']):
     allref=vals[[names.index(x) for x in random],ci]
     for name in original:
      target=vals[names.index(name),ci];ref=vals[[names.index(x) for x in matched[name]],ci];enough=len(ref)>=rule['minimum_comparable_references']
      ranks.append(dict(prompt=pr,nutrient=nut,window=window,contrast=c,ROI=name,value=float(target),reference_n=20,unmatched_fraction_below=float(np.mean(allref<target)),matched_n=len(ref),matched_status='sufficient_descriptive_only' if enough else 'insufficient_no_rank_claim',matched_fraction_below=float(np.mean(ref<target)) if enough else '',matched_median=float(np.median(ref)) if enough else '',LOSO=metrics[name]['LOSO'],median_fixed_voxels=metrics[name]['median_fixed_voxels']))
 write_csv(out/'individual_HCP_reference_positions.csv',ranks)
 frozen=read(ROOT/cfg['input_preparation']/'random_parcels.json');nine=frozen['nine_parcel_reference_groups'];assert len(nine)==1000
 # Each random nine-parcel group retains participant pairing and all layers.
 nineco=np.stack([co[:,:,[names.index(x) for x in group]].mean(axis=2) for group in nine],axis=2)
 np.save(out/'random_nine_group_contrasts.npy',nineco)
 ngmetrics=[]
 for j,pp in enumerate(nine):
  vv=[np.mean([r['fixed_voxels'] for r in subject_metrics if r['ROI'] in pp and r['subject_id']==sub]) for sub in subjects]
  ngmetrics.append(dict(group_id=j,LOSO=float(np.mean([metrics[x]['LOSO'] for x in pp])),median_fixed_voxels=float(np.median(vv)),mean_coverage=float(np.mean([metrics[x]['mean_coverage'] for x in pp]))))
 write_csv(out/'random_nine_group_quality.csv',ngmetrics)
 target=metrics['OFC_nine'];eligible=[r['group_id'] for r in ngmetrics if abs(r['LOSO']-target['LOSO'])<=rule['LOSO_absolute_caliper'] and .5<=r['median_fixed_voxels']/target['median_fixed_voxels']<=2]
 nine_positions=[]
 for pi,pr in enumerate(prompts):
  for nj,nut in enumerate(nutrients):
   for w,ls in ix['windows'].items():
    rv=nineco[pi,nj][:,:,np.array(ls)-1,:].mean(axis=(1,2));tv=co[pi,nj,names.index('OFC_nine')][:,np.array(ls)-1,:].mean(axis=(0,1))
    for ci,c in enumerate(ix['contrasts']):
     nine_positions.append(dict(prompt=pr,nutrient=nut,window=w,contrast=c,OFC_nine=float(tv[ci]),reference_groups=1000,unmatched_fraction_below=float(np.mean(rv[:,ci]<tv[ci])),matched_groups=len(eligible),matched_fraction_below=float(np.mean(rv[eligible,ci]<tv[ci])) if len(eligible)>=5 else '',matched_status='sufficient_descriptive_only' if len(eligible)>=5 else 'insufficient_no_rank_claim',dependent_groups_not_null=True))
 write_csv(out/'OFC_nine_reference_positions.csv',nine_positions)
 (out/'reference_index.json').write_text(json.dumps(dict(nine_parcel_groups=nine,matched_nine_group_ids=eligible,nine_contrast_axes=['prompt','nutrient','group','participant','layer','contrast'],matching_calipers=rule,interpretation='descriptive overlapping groups from fixed20; no p values'),indent=2))
 # Ranges are implementation spread, never confidence intervals.
 spread=[]
 for pi,pr in enumerate(prompts):
  for nj,nut in enumerate(nutrients):
   for ri,name in enumerate(names):
    for w,ls in ix['windows'].items():
     inds=np.array(ls)-1;n=ns[pi,ri][:,inds].mean(axis=(0,1));q=vtp[pi,nj,ri][:,inds,2:].mean(axis=(0,1));v=float(vtp[pi,nj,ri][:,inds,0].mean());t=float(vtp[pi,nj,ri][:,inds,1].mean())
     spread.append(dict(prompt=pr,nutrient=nut,ROI=name,window=w,V=v,T=t,N1=float(n[0]),N2=float(n[1]),N3=float(n[2]),N_mean=float(n.mean()),N_min=float(n.min()),N_max=float(n.max()),P_mean=float(q.mean()),P_min=float(q.min()),P_max=float(q.max()),T_minus_N_min=float(t-n.max()),T_minus_N_max=float(t-n.min()),T_minus_P_min=float(t-q.max()),T_minus_P_max=float(t-q.min())))
 write_csv(out/'implementation_ranges.csv',spread)
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 colors=['#8b8b8b','#d98e04','#7045a4','#15817c'];nlabels=['Protein','Fat','Carbohydrate','Fibre'];layers=np.arange(1,33)
 def finish(fig,name):
  fig.savefig(figdir/(name+'.png'),dpi=140,bbox_inches='tight');fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 def basic(ax):
  ax.axhline(0,color='#aaaaaa',lw=.6);ax.set_xlim(1,32);ax.set_xticks([1,8,16,24,32]);ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
 region=['OFC_nine','higher_visual','early_intermediate_visual']
 for pi,pr in enumerate(prompts):
  fig,axs=plt.subplots(3,4,figsize=(16,10),sharex=True)
  for rj,r in enumerate(region):
   ri=names.index(r)
   for nj in range(4):
    ax=axs[rj,nj];s=vtp[pi,nj,ri].mean(0);n=ns[pi,ri].mean(0)
    for color,label,y in zip(colors,['V','mean N','T','mean P'],[s[:,0],n.mean(1),s[:,1],s[:,2:].mean(1)]):ax.plot(layers,y,color=color,label=label)
    ax.fill_between(layers,n.min(1),n.max(1),color=colors[1],alpha=.16);ax.fill_between(layers,s[:,2:].min(1),s[:,2:].max(1),color=colors[3],alpha=.12)
    basic(ax);ax.set_title(nlabels[nj]+' | '+r);ax.set_ylabel('RSA Spearman rho')
  axs[0,0].legend(ncol=2,fontsize=8);fig.suptitle(f'Prompt {pr} | Absolute alignment | 29 participant means\nShading: 3 N / 5 P implementation range, NOT a confidence interval');fig.supxlabel('Decoder layer');fig.tight_layout(rect=(0,0,1,.94));finish(fig,'absolute_'+pr)
  fig,axs=plt.subplots(4,3,figsize=(15,12),sharex=True)
  for nj in range(4):
   for ci,c in enumerate(['N - V','T - N','T - meanP']):
    ax=axs[nj,ci];idx=[0,1,3][ci]
    for r,color in zip(region,['#7045a4','#15817c','#cf811a']):ax.plot(layers,co[pi,nj,names.index(r),:,:,idx].mean(0),color=color,label=r)
    basic(ax);ax.set_title(nlabels[nj]+' | '+c)
  axs[0,0].legend(fontsize=8);fig.suptitle(f'Prompt {pr} | Control contrasts across all 32 layers\nN-V is shared across nutrients; repeated panels are not independent replications');fig.supxlabel('Decoder layer');fig.supylabel('Change in RSA');fig.tight_layout(rect=(.02,0,1,.94));finish(fig,'contrasts_'+pr)
  # All 41 ROIs: 18 original, 20 random, 3 legacy; no outcome selection.
  vals=co[pi,:,:41].mean(axis=2);bound=np.max(abs(vals))
  fig,axs=plt.subplots(3,4,figsize=(17,19),sharex=True,sharey=True)
  for nj in range(4):
   for row,ci in enumerate([0,1,3]):
    ax=axs[row,nj];im=ax.imshow(vals[nj,:,:,ci],aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound,extent=[.5,32.5,40.5,-.5]);ax.set_yticks(range(41),names[:41],fontsize=7);ax.set_xticks([1,16,32]);ax.axhline(17.5,color='black',lw=.8);ax.axhline(37.5,color='black',lw=.8);ax.set_title(nlabels[nj]+' | '+ix['contrasts'][ci])
  fig.suptitle(f'Prompt {pr} | 18 original HCP + 20 random HCP + 3 legacy references\nCommon color scale; hemisphere RSA averaged first; legacy has a different ROI definition');fig.tight_layout(rect=(0,0,.95,.96));cb=fig.add_axes([.96,.2,.01,.55]);fig.colorbar(im,cax=cb,label='Mean change in RSA');finish(fig,'all41_heatmap_'+pr)
  for w,ls in ix['windows'].items():
   means=co[pi][:,:, :,np.array(ls)-1,:].mean(axis=(2,3))
   fig,axs=plt.subplots(2,4,figsize=(16,8))
   for nj in range(4):
    for row,ci in enumerate([2,1]):
     ax=axs[row,nj]
     for name in parcels:
      israndom=name in random;isofc=name in groups['OFC_nine'];color='#a6a6a6' if israndom else ('#7045a4' if isofc else '#15817c');ax.scatter(metrics[name]['LOSO'],means[nj,names.index(name),ci],s=20 if israndom else 34,c=color,marker='o' if israndom else '^',alpha=.85)
     for name in ['lOFC','mOFC','vmPFC']:
      ax.scatter(metrics[name]['LOSO'],means[nj,names.index(name),ci],marker='x',c='#c66c17',s=40);ax.annotate(name,(metrics[name]['LOSO'],means[nj,names.index(name),ci]),fontsize=6,xytext=(3,3),textcoords='offset points')
     ax.axhline(0,color='gray',lw=.5);ax.set_title(nlabels[nj]+' | '+ix['contrasts'][ci]);ax.set_xlabel('LOSO consistency (not test-retest)');ax.set_ylabel('Mean RSA change')
   fig.suptitle(f'{pr} | {w} | Grey: random20; purple: OFC9; teal: other original; orange: legacy\nLOSO is descriptive context; legacy crosses are not used in matched parcel ranks');fig.tight_layout(rect=(0,0,1,.93));finish(fig,f'consistency_scatter_{pr}_{w}')
 # Display all fixed nine-parcel reference groups; overlap is descriptive, not an empirical null.
 for w,ls in ix['windows'].items():
  fig,axs=plt.subplots(2,4,figsize=(16,8))
  for nj in range(4):
   for row,ci in enumerate([2,1]):
    ax=axs[row,nj]
    for pi,pr in enumerate(prompts):
     v=nineco[pi,nj][:,:,np.array(ls)-1,ci].mean(axis=(1,2));target=co[pi,nj,names.index('OFC_nine')][:,np.array(ls)-1,ci].mean()
     color=['#7045a4','#15817c'][pi];ax.hist(v,bins=25,density=True,histtype='step',color=color,label=pr+' random9 groups');ax.axvline(target,color=color,ls='--',lw=2,label=pr+' OFC9')
    ax.set_title(nlabels[nj]+' | '+ix['contrasts'][ci]);ax.set_xlabel('Mean RSA change');ax.set_ylabel('Density of descriptive groups')
  axs[0,0].legend(fontsize=7);fig.suptitle(f'{w} | OFC9 against 1,000 frozen nine-parcel summaries from random20\nGroups overlap and are not independent; this is NOT a null distribution or significance test.');fig.tight_layout(rect=(0,0,1,.93));finish(fig,'nine_group_reference_'+w)
 # Participant distributions, both windows and prompts, all four nutrients.
 for w,ls in ix['windows'].items():
  fig,axs=plt.subplots(3,4,figsize=(16,10))
  for rj,r in enumerate(['OFC_nine','lOFC','vmPFC']):
   for nj in range(4):
    ax=axs[rj,nj]
    for k,(pi,ci) in enumerate([(0,0),(1,0),(0,1),(1,1),(0,3),(1,3)]):
     v=co[pi,nj,names.index(r),:,np.array(ls)-1,ci].mean(axis=0)
     assert v.shape==(29,)
     ax.scatter(k+np.linspace(-.13,.13,29),v,s=10,alpha=.65,c=colors[[1,1,2,2,3,3][k]]);ax.plot([k-.18,k+.18],[v.mean()]*2,color='black',lw=2)
    ax.axhline(0,color='gray',lw=.6);ax.set_xticks(range(6),['A NV','B NV','A TN','B TN','A TP','B TP'],rotation=35,fontsize=7);ax.set_title(nlabels[nj]+' | '+r);ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
  fig.suptitle(f'Participant distributions | {w} | 29 dots per condition, black mean\nNV=N-V; TN=T-meanN; TP=T-meanP. No confidence intervals or significance tests.');fig.tight_layout(rect=(0,0,1,.94));finish(fig,'participants_'+w)
 audit=dict(status='summaries_and_plots_complete_visual_review_pending',created=datetime.now().astimezone().isoformat(),RSA_audit_sha256=sha(p/'audit.json'),script_sha256=sha(__file__),matched_reference_counts={k:len(v) for k,v in matched.items()},matched_nine_groups=len(eligible),inference_run=False,versions=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),output_sha256={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()})
 (out/'audit.json').write_text(json.dumps(audit,indent=2));print('Reference summaries and plots complete',flush=True)
if __name__=='__main__':main()
