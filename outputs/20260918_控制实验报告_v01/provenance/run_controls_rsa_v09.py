"""Frozen v09 descriptive control RSA. Preserve all original V/T/P scores."""
import csv,json,platform,shutil,subprocess,time
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist,squareform
from threadpoolctl import threadpool_limits
from prepare_roi import sha,write_csv
from rsa_core import ranked_unit,score,validate,correlation_rdm
from run_hcp_rsa import checked
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def main():
 cfgp=ROOT/'configs/controls_v09_execution_v01.json';cfg=read(cfgp)
 out=ROOT/cfg['RSA_output'];
 for pr in ['A','B']:assert (ROOT/f'outputs/20260918_无关文本模型缓存_v01/controls_{pr}_matrix_v01/audit.json').exists(), 'Local model transfer incomplete'
 out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'script_snapshot.py')
 inputs={str(p):sha(p) for p in [cfgp,Path(__file__),ROOT/'scripts/rsa_core.py',ROOT/'scripts/run_hcp_rsa.py']}
 old=ROOT/'outputs/20260917_HCP_RSA_v01';oi=read(old/'index.json');oa=read(old/'audit.json')
 oldbrain=ROOT/'outputs/20260917_HCP原始模式_v01';newbrain=ROOT/cfg['patterns_output'];legacy=ROOT/'outputs/20260917_四营养A_RSA_v01'
 sources=[oldbrain,legacy,newbrain];audits={p:read(p/'audit.json') for p in sources}
 for p,a in audits.items():
  assert a['status']=='complete';inputs[str(p/'audit.json')]=sha(p/'audit.json')
 ni=read(newbrain/'index.json');assert ni['image_ids']==oi['image_ids'] and ni['subject_ids']==oi['subject_ids']
 subjects=oi['subject_ids'];images=oi['image_ids'];nutrients=oi['nutrients'];prompts=['A','B'];conditions=oi['conditions']
 oldnames=oi['unilateral_and_legacy_ROIs'];newnames=ni['ROIs'];names=oldnames+newnames;assert len(names)==79
 masks={};brains={};grouped={};qc=[];errors=[];verification=validate()
 for si,sub in enumerate(subjects):
  gm=None;ds=[]
  for ri,name in enumerate(names):
   root=oldbrain if ri<36 else (legacy if ri<39 else newbrain)
   base=root/name/sub if root!=legacy else root/name/'brain'/sub
   g=np.asarray(checked(base/'global_image_rows.npy',root,audits[root],inputs))
   if gm is None:gm=g.copy()
   else:assert np.array_equal(gm,g),'Masks differ: must explicitly intersect before scoring'
   d=checked(base/'RDM.npy',root,audits[root],inputs);assert d.shape==(len(g),len(g)) and np.isfinite(d).all()
   ds.append(d)
  masks[sub]=gm;brains[sub]=ds;tr=np.tril_indices(len(gm),-1)
  ranked=np.stack([ranked_unit(d[tr]) for d in ds])
  key=tuple(gm);grouped.setdefault(key,{'subjects':[],'blocks':[]});grouped[key]['subjects'].append(si);grouped[key]['blocks'].append(ranked)
  for q in ['session','run']:
   d=checked(newbrain/'trials'/sub/(q+'_RDM.npy'),newbrain,audits[newbrain],inputs)
   vals=ranked[39:]@ranked_unit(d[tr])
   qc.extend(dict(subject_id=sub,ROI=name,metric=q+'_RDM_rho',value=float(v),images=len(gm),pairs=len(tr[0])) for name,v in zip(newnames,vals))
 for gk,g in grouped.items():
  gi=np.array(gk);tr=np.tril_indices(len(gi),-1);g['i']=gi[tr[0]];g['j']=gi[tr[1]]
  g['matrix']=np.concatenate(g.pop('blocks'));g['new_matrix']=g['matrix'].reshape(len(g['subjects']),79,-1)[:,39:].reshape(-1,len(tr[0])).copy()
 print('79 brain ROIs ranked; common participant masks confirmed',flush=True)
 for ri,name in enumerate(newnames,39):
  total=np.zeros((401,401));count=np.zeros((401,401),int)
  for sub in subjects:
   g=masks[sub];total[np.ix_(g,g)]+=brains[sub][ri];count[np.ix_(g,g)]+=1
  for sub in subjects:
   g=masks[sub];d=brains[sub][ri];n=count[np.ix_(g,g)]-1;tr=np.tril_indices(len(g),-1);assert np.all(n[tr]>0)
   other=(total[np.ix_(g,g)]-d)/np.maximum(n,1)
   qc.append(dict(subject_id=sub,ROI=name,metric='LOSO_group_RDM_rho_not_test_retest',value=score(d[tr],other[tr]),images=len(g),pairs=len(tr[0])))
 with (old/'acquisition_and_consistency_QC.csv').open() as f:qc=list(csv.DictReader(f))+qc
 inputs[str(old/'acquisition_and_consistency_QC.csv')]=sha(old/'acquisition_and_consistency_QC.csv')
 write_csv(out/'acquisition_and_consistency_QC.csv',qc)
 def evaluate(rdms,newonly):
  nr=40 if newonly else 79;result=np.empty((nr,29,32));offset=39 if newonly else 0
  for li,d in enumerate(rdms):
   for g in grouped.values():
    z=ranked_unit(d[g['i'],g['j']]);v=g['new_matrix' if newonly else 'matrix']@z
    for j,si in enumerate(g['subjects']):result[:,si,li]=v[j*nr:(j+1)*nr]
   if li in [0,15,31]:
    for si,ri in [(0,0),(13,17),(28,nr-1)]:
     sub=subjects[si];gi=masks[sub];tr=np.tril_indices(len(gi),-1)
     direct=spearmanr(d[gi[tr[0]],gi[tr[1]]],brains[sub][ri+offset][tr]).statistic
     er=abs(direct-result[ri,si,li]);assert er<1e-12;errors.append(float(er))
  return result
 previous=checked(old/'unilateral_and_legacy_scores.npy',old,oa,inputs)
 vtp=np.empty((2,4,79,29,32,7));vtp[:,:,:39]=previous
 ns=np.empty((2,79,29,32,3));matrixroot=ROOT/'outputs/20260918_无关文本模型缓存_v01'
 for pi,pr in enumerate(prompts):
  mr=matrixroot/f'controls_{pr}_matrix_v01';ma=read(mr/'audit.json');assert ma['status']=='complete_verified' and ma['common_image_ids']==images
  inputs[str(mr/'audit.json')]=sha(mr/'audit.json');rdmdir=out/pr;rdmdir.mkdir()
  for ci,condition in enumerate(cfg['conditions_N']):
   grp=next(e for e in ma['groups'] if e['condition']==condition)
   for rel,digest in grp['files'].items():assert sha(mr/rel)==digest;inputs[str(mr/rel)]=digest
   hidden=np.load(mr/f'{pr}_unrelated_text_{condition}'/'hidden.npy',mmap_mode='r')
   ds=np.stack([correlation_rdm(h) for h in hidden]);np.save(rdmdir/f'{condition}_RDM.npy',ds)
   for li in [0,15,31]:assert np.max(abs(ds[li,:20,:20]-squareform(pdist(hidden[li,:20].astype(float),'correlation'))))<1e-12
   ns[pi,:,:,:,ci]=evaluate(ds,False);print(f'{pr} {condition}: 79 ROI curves complete',flush=True)
  raw=ROOT/'outputs/20260917_raw_RSA_v01';ra=read(raw/'audit.json');rids=read(raw/'image_index.json')['image_ids'];take=np.array([rids.index(x) for x in images])
  v=np.asarray(checked(raw/pr/'V_RDM.npy',raw,ra,inputs))[:,take][:,:,take];vs=evaluate(v,True)
  modelroot=ROOT/f'outputs/20260917_四营养{pr}_RSA_v01';a=read(modelroot/'audit.json')
  for nj,nut in enumerate(nutrients):
   vtp[pi,nj,39:,:,: ,0]=vs
   for ci,cond in enumerate(conditions[1:],1):
    ds=checked(modelroot/nut/(cond+'_RDM.npy'),modelroot,a,inputs)
    vtp[pi,nj,39:,:,:,ci]=evaluate(ds,True)
   print(f'{pr} {nut}: random 40 ROI V/T/P complete',flush=True)
  np.save(out/f'{pr}_N_unilateral_and_legacy_scores.npy',ns[pi]);np.save(out/f'{pr}_VTP_unilateral_and_legacy_scores.npy',vtp[pi])
 assert np.array_equal(vtp[:,:,:39],previous) and np.isfinite(vtp).all() and np.isfinite(ns).all()
 parcels=oi['series'][:18]+cfg['HCP_groups']['random_reference'];groups=oi['groups'];series=parcels+['lOFC','mOFC','vmPFC']+list(groups)
 def aggregate(arr,axis):
  pieces=[(np.take(arr,names.index('L_'+p),axis=axis)+np.take(arr,names.index('R_'+p),axis=axis))/2 for p in parcels]
  pieces += [np.take(arr,names.index(p),axis=axis) for p in ['lOFC','mOFC','vmPFC']]
  pieces += [np.mean([pieces[parcels.index(p)] for p in pp],axis=0) for pp in groups.values()]
  return np.stack(pieces,axis=axis)
 bs=aggregate(vtp,2);bn=aggregate(ns,1);nm=bn.mean(axis=-1);V=bs[...,0];T=bs[...,1]
 contrast=np.stack([np.broadcast_to(nm[:,None],V.shape)-V,T-nm[:,None],T-V,T-bs[...,2:].mean(axis=-1)],axis=-1)
 assert np.max(abs(contrast[...,0]+contrast[...,1]-contrast[...,2]))<1e-15
 for k,n in enumerate(oi['series']):assert np.array_equal(bs[:,:,series.index(n)],checked(old/'bilateral_and_reference_and_group_scores.npy',old,oa,inputs)[:,:,k])
 for name,arr in [('unilateral_VTP_scores',vtp),('unilateral_N_scores',ns),('series_VTP_scores',bs),('series_N_scores',bn),('series_contrasts',contrast)]:np.save(out/(name+'.npy'),arr)
 regional=np.stack([contrast[:,:,series.index('OFC_nine')]-contrast[:,:,series.index(g)] for g in ['higher_visual','early_intermediate_visual']],axis=2)
 np.save(out/'regional_paired_contrasts.npy',regional)
 summary=[];participants=[];curves=[];absolute=[]
 for pi,pr in enumerate(prompts):
  for nj,nut in enumerate(nutrients):
   for ri,name in enumerate(series+['OFC_minus_higher_visual','OFC_minus_early_intermediate_visual']):
    data=contrast[pi,nj,ri] if ri<len(series) else regional[pi,nj,ri-len(series)]
    for ci,c in enumerate(cfg['contrasts']):
     x=data[...,ci]
     for li in range(32):curves.append(dict(prompt=pr,nutrient=nut,ROI=name,contrast=c,layer=li+1,mean=float(x[:,li].mean()),subject_q25=float(np.quantile(x[:,li],.25)),subject_q75=float(np.quantile(x[:,li],.75))))
     for w,ls in cfg['layer_windows'].items():
      av=x[:,np.array(ls)-1].mean(axis=1)
      summary.append(dict(prompt=pr,nutrient=nut,ROI=name,contrast=c,window=w,mean=float(av.mean()),median=float(np.median(av)),subject_q25=float(np.quantile(av,.25)),subject_q75=float(np.quantile(av,.75)),subjects_positive=int((av>0).sum()),participants=29))
      participants.extend(dict(prompt=pr,nutrient=nut,ROI=name,contrast=c,window=w,subject_id=sub,value=float(val)) for sub,val in zip(subjects,av))
    if ri<len(series):
     mm=bs[pi,nj,ri].mean(axis=0);nn=bn[pi,ri].mean(axis=0)
     for li in range(32):absolute.append(dict(prompt=pr,nutrient=nut,ROI=name,layer=li+1,V=float(mm[li,0]),T=float(mm[li,1]),P_mean=float(mm[li,2:].mean()),P_min=float(mm[li,2:].min()),P_max=float(mm[li,2:].max()),N_mean=float(nn[li].mean()),N_min=float(nn[li].min()),N_max=float(nn[li].max())))
 for fn,rr in [('window_summary',summary),('participant_window_summary',participants),('contrast_curves',curves),('absolute_curves',absolute)]:write_csv(out/(fn+'.csv'),rr)
 index=dict(prompts=prompts,nutrients=nutrients,unilateral_ROIs=names,series=series,parcels=parcels,random_parcels=cfg['HCP_groups']['random_reference'],original_parcels=oi['series'][:18],groups=groups,subject_ids=subjects,image_ids=images,layers=list(range(1,33)),VTP_conditions=conditions,N_conditions=cfg['conditions_N'],contrasts=cfg['contrasts'],VTP_axes=['prompt','nutrient','ROI','participant','layer','condition'],N_axes=['prompt','ROI','participant','layer','N_implementation'],contrast_axes=['prompt','nutrient','ROI','participant','layer','contrast'],windows=cfg['layer_windows'],all_masks_identical_within_person=True)
 (out/'index.json').write_text(json.dumps(index,indent=2))
 audit=dict(status='complete',created=datetime.now().astimezone().isoformat(),participants=29,images=401,new_brain_RDMs=1160,new_model_RDMs=192,original_VTP_exactly_preserved=True,all_pair_masks_identical=True,new_exclusions=[],independent_Spearman_max_error=max(errors),method_validation=verification,N_and_P_aggregation='score_mean_not_RDM_mean',extra_brain_preprocessing=[],inference_run=False,input_sha256=inputs,git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
 (out/'audit.json').write_text(json.dumps(audit,indent=2));print('Control RSA complete',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
