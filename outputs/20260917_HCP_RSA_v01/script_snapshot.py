"""Descriptive HCP RSA, with all hemisphere/parcel scores and unchanged legacy references."""
import csv, gzip, json, platform, shutil, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from prepare_roi import sha, write_csv
from extract_rsa_patterns import rows
from rsa_core import ranked_unit, score, validate

ROOT=Path(__file__).resolve().parents[1]

def checked(p, root, audit, inputs):
    digest=sha(p);assert digest==audit['output_sha256'][str(p.relative_to(root))],str(p)
    inputs[str(p)]=digest
    return np.load(p,mmap_mode='r')

def main():
    cfgp=ROOT/'configs/hcp_regional_execution_v01.json';cfg=json.loads(cfgp.read_text())
    source=ROOT/cfg['patterns_output'];pa=json.loads((source/'audit.json').read_text());assert pa['status']=='complete'
    ix=json.loads((source/'index.json').read_text());names=ix['ROIs'];subjects=ix['subject_ids'];images=ix['image_ids']
    out=ROOT/cfg['RSA_output'];assert not out.exists();out.mkdir(parents=True)
    shutil.copy2(__file__,out/'script_snapshot.py')
    inputs={str(p):sha(p) for p in [cfgp,source/'audit.json',source/'index.json',Path(__file__),ROOT/'scripts/rsa_core.py']}
    roots={pr:ROOT/cfg['legacy_references'][pr+'_cache'] for pr in ['A','B']}
    audits={pr:json.loads((p/'audit.json').read_text()) for pr,p in roots.items()}
    oldidx={pr:json.loads((p/'index.json').read_text()) for pr,p in roots.items()}
    legacy=oldidx['A']['ROIs'];allnames=names+legacy
    for pr,p in roots.items():
        inputs[str(p/'audit.json')]=sha(p/'audit.json');inputs[str(p/'index.json')]=sha(p/'index.json')
        assert oldidx[pr]['image_ids']==images and oldidx[pr]['subject_ids']==subjects
    nutrients=oldidx['A']['nutrients'];conditions=oldidx['A']['conditions'];N=len(images)
    masks={};brains={};grouped={};qc=[];verification=validate();errors=[]
    # Every actual HCP pattern passed. Assert all sides and old references share this exact per-person pair mask.
    for si,sub in enumerate(subjects):
        gm=None;bs=[]
        for name in allnames:
            root=source if name in names else roots['A']
            audit=pa if name in names else audits['A']
            base=root/name/sub if name in names else root/name/'brain'/sub
            g=np.asarray(checked(base/'global_image_rows.npy',root,audit,inputs))
            if gm is None:gm=g.copy()
            else:assert np.array_equal(gm,g),'Nonidentical image mask: must intersect and rerank before proceeding'
            d=checked(base/'RDM.npy',root,audit,inputs);assert d.shape==(len(g),len(g))
            assert np.isfinite(d).all() and np.allclose(d,d.T,rtol=0,atol=1e-12)
            bs.append(d)
        masks[sub]=gm;brains[sub]=bs
        tri=np.tril_indices(len(gm),-1)
        ranked=np.stack([ranked_unit(d[tri]) for d in bs])
        key=tuple(gm);grouped.setdefault(key,{'subjects':[],'blocks':[]})
        grouped[key]['subjects'].append(si);grouped[key]['blocks'].append(ranked)
        for q in ['session','run']:
            p=source/'trials'/sub/(q+'_RDM.npy');d=checked(p,source,pa,inputs)
            values=ranked@ranked_unit(d[tri])
            for ri,name in enumerate(allnames):qc.append(dict(subject_id=sub,ROI=name,metric=q+'_RDM_rho',value=float(values[ri]),images=len(gm),pairs=len(tri[0])))
    for key,g in grouped.items():
        gi=np.array(key);tri=np.tril_indices(len(gi),-1)
        g['pair_i']=gi[tri[0]];g['pair_j']=gi[tri[1]];g['matrix']=np.concatenate(g.pop('blocks'),axis=0)
    print('Brain rank vectors ready; all 39 ROIs share each participant pair mask',flush=True)
    # Leave-one-subject-out consistency with a pairwise available group mean; descriptive, not repeated-measure reliability.
    for ri,name in enumerate(allnames):
        total=np.zeros((N,N));count=np.zeros((N,N),int)
        for sub in subjects:
            g=masks[sub];total[np.ix_(g,g)]+=brains[sub][ri];count[np.ix_(g,g)]+=1
        for sub in subjects:
            g=masks[sub];d=brains[sub][ri];n=count[np.ix_(g,g)]-1;tri=np.tril_indices(len(g),-1)
            assert np.all(n[tri]>0)
            other=(total[np.ix_(g,g)]-d)/np.maximum(n,1)
            qc.append(dict(subject_id=sub,ROI=name,metric='LOSO_group_RDM_rho_not_test_retest',value=score(d[tri],other[tri]),images=len(g),pairs=len(tri[0])))
    write_csv(out/'acquisition_and_consistency_QC.csv',qc)
    old=ROOT/'outputs/20260917_raw_RSA_v01';oa=json.loads((old/'audit.json').read_text())
    oi=json.loads((old/'image_index.json').read_text())
    old_images=oi['image_ids'] if isinstance(oi,dict) else oi
    take=np.array([old_images.index(x) for x in images])
    inputs[str(old/'audit.json')]=sha(old/'audit.json');inputs[str(old/'image_index.json')]=sha(old/'image_index.json')
    # axes: prompt, nutrient, unilateral/legacy ROI, subject, layer, condition
    scores=np.full((2,4,len(allnames),29,32,7),np.nan)
    legacy_max_error=0.
    for pi,pr in enumerate(['A','B']):
        def evaluate(rdms, expected=None):
            result=np.empty((len(allnames),29,32))
            for li in range(32):
                d=rdms[li]
                for g in grouped.values():
                    z=ranked_unit(d[g['pair_i'],g['pair_j']]);v=g['matrix']@z
                    for j,si in enumerate(g['subjects']):result[:,si,li]=v[j*len(allnames):(j+1)*len(allnames)]
                if li in [0,15,31]:
                    for si,ri in [(0,0),(7,17),(28,35)]:
                        sub=subjects[si];gi=masks[sub];tr=np.tril_indices(len(gi),-1)
                        direct=spearmanr(d[gi[tr[0]],gi[tr[1]]],brains[sub][ri][tr]).statistic
                        er=abs(direct-result[ri,si,li]);assert er<1e-12;errors.append(float(er))
            if expected is not None:assert np.allclose(result[-3:],expected,rtol=0,atol=2e-12)
            return result
        v=np.asarray(checked(old/pr/'V_RDM.npy',old,oa,inputs))[:,take][:,:,take]
        v_expected=checked(roots[pr]/'shared_V_neural_scores.npy',roots[pr],audits[pr],inputs)
        vscore=evaluate(v,v_expected)
        legacy_max_error=max(legacy_max_error,float(np.max(abs(vscore[-3:]-v_expected))))
        for ni,nut in enumerate(nutrients):
            expected=checked(roots[pr]/nut/'neural_scores.npy',roots[pr],audits[pr],inputs)
            scores[pi,ni,:,:,:,0]=vscore
            for ci,cond in enumerate(conditions[1:],1):
                d=checked(roots[pr]/nut/(cond+'_RDM.npy'),roots[pr],audits[pr],inputs)
                result=evaluate(d,expected[:,:,:,ci]);scores[pi,ni,:,:,:,ci]=result
                legacy_max_error=max(legacy_max_error,float(np.max(abs(result[-3:]-expected[:,:,:,ci]))))
                print(f'{pr} {nut} {cond}: 39 ROI RSA curves complete',flush=True)
            np.save(out/f'{pr}_{nut}_unilateral_and_legacy_scores.npy',scores[pi,ni])
    assert np.isfinite(scores).all()
    parcels=[p for g in cfg['HCP_groups'].values() for p in g]
    bilateral=np.stack([(scores[:,:,names.index('L_'+p)]+scores[:,:,names.index('R_'+p)])/2 for p in parcels],axis=2)
    summary_groups={k:v for k,v in cfg['HCP_groups'].items() if k!='medial_prefrontal_references'}
    summary_groups['OFC_nine']=cfg['HCP_groups']['lateral_OFC']+cfg['HCP_groups']['medial_OFC']
    series_names=parcels+legacy+list(summary_groups)
    combined=np.concatenate([bilateral,scores[:,:,-3:],np.stack([np.mean(bilateral[:,:,[parcels.index(p) for p in pp]],axis=2) for pp in summary_groups.values()],axis=2)],axis=2)
    contrasts=np.stack([combined[...,1]-combined[...,0],combined[...,1]-combined[...,2:].mean(axis=-1)],axis=-1)
    np.save(out/'bilateral_and_reference_and_group_scores.npy',combined)
    np.save(out/'bilateral_and_reference_and_group_contrasts.npy',contrasts)
    np.save(out/'unilateral_and_legacy_scores.npy',scores)
    np.save(out/'unilateral_and_legacy_contrasts.npy',np.stack([scores[...,1]-scores[...,0],scores[...,1]-scores[...,2:].mean(axis=-1)],axis=-1))
    regional_names=['OFC_minus_higher_visual','OFC_minus_early_intermediate_visual']
    regional=np.stack([contrasts[:,:,series_names.index('OFC_nine')]-contrasts[:,:,series_names.index(g)] for g in ['higher_visual','early_intermediate_visual']],axis=2)
    np.save(out/'regional_paired_contrasts.npy',regional)
    # Full scores exported compressed; arrays are the primary lossless record.
    with gzip.open(out/'all_subject_scores.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(['prompt','nutrient','ROI','subject_id','layer','condition','RSA'])
        for pi,pr in enumerate(['A','B']):
            for ni,nut in enumerate(nutrients):
                for ri,name in enumerate(allnames+series_names[:18]+list(summary_groups)):
                    data=scores[pi,ni,ri] if ri<len(allnames) else combined[pi,ni,series_names.index(name)]
                    for si,sub in enumerate(subjects):
                        for li in range(32):
                            for ci,c in enumerate(conditions):w.writerow([pr,nut,name,sub,li+1,c,float(data[si,li,ci])])
    curve=[];means=[];participant=[]
    for pi,pr in enumerate(['A','B']):
        for ni,nut in enumerate(nutrients):
            for ri,name in enumerate(series_names+regional_names):
                data=contrasts[pi,ni,ri] if ri<len(series_names) else regional[pi,ni,ri-len(series_names)]
                for ci,c in enumerate(['T-V','T-meanP']):
                    a=data[:,:,ci]
                    for li in range(32):curve.append(dict(prompt=pr,nutrient=nut,ROI=name,contrast=c,layer=li+1,participants=29,mean=float(a[:,li].mean()),median=float(np.median(a[:,li])),participant_q25=float(np.quantile(a[:,li],.25)),participant_q75=float(np.quantile(a[:,li],.75)),participants_positive=int((a[:,li]>0).sum())))
                    av=a.mean(axis=1)
                    means.append(dict(prompt=pr,nutrient=nut,ROI=name,contrast=c,participants=29,mean_all32=float(av.mean()),median_subject_mean=float(np.median(av)),participant_q25=float(np.quantile(av,.25)),participant_q75=float(np.quantile(av,.75)),participants_positive=int((av>0).sum())))
                    for si,sub in enumerate(subjects):participant.append(dict(prompt=pr,nutrient=nut,ROI=name,contrast=c,subject_id=sub,mean_all32=float(av[si])))
    write_csv(out/'group_contrast_curves.csv',curve);write_csv(out/'all_layer_summary.csv',means);write_csv(out/'participant_all_layer_summary.csv',participant)
    absolute=[]
    for pi,pr in enumerate(['A','B']):
        for ni,nut in enumerate(nutrients):
            for ri,name in enumerate(series_names):
                avg=combined[pi,ni,ri].mean(axis=0)
                for li in range(32):absolute.append(dict(prompt=pr,nutrient=nut,ROI=name,layer=li+1,V=float(avg[li,0]),T=float(avg[li,1]),P_mean=float(avg[li,2:].mean()),P_min=float(avg[li,2:].min()),P_max=float(avg[li,2:].max())))
    write_csv(out/'absolute_alignment_curves.csv',absolute)
    index=dict(prompts=['A','B'],nutrients=nutrients,unilateral_and_legacy_ROIs=allnames,series=series_names,regional_comparisons=regional_names,subject_ids=subjects,image_ids=images,layers=list(range(1,33)),conditions=conditions,contrasts=['T-V','T-meanP'],array_axes=['prompt','nutrient','ROI_or_comparison','subject','layer','condition_or_contrast'],groups=summary_groups,common_pair_masks_all_ROIs=True,missing_values=0)
    (out/'index.json').write_text(json.dumps(index,indent=2))
    audit=dict(status='complete',created=datetime.now().astimezone().isoformat(),protocol=cfgp.name,raw_beta=True,extra_preprocessing=[],descriptive_only=True,inference_run=False,
      seed=None,participants=29,common_images=401,HCP_unilateral=36,HCP_bilateral=18,legacy_references=3,all_pair_masks_identical_within_participant=True,
      individual_missing_trials_not_imputed=True,new_exclusions=[],legacy_scores_independently_reproduced_max_error=legacy_max_error,
      independent_Spearman_spotcheck_max_error=max(errors),method_validation=verification,
      P_aggregation='mean_of_scores_not_RDM_or_hidden_mean',OFC_aggregation='equal_nine_parcel_mean_after_equal_hemisphere_score_mean',
      QC='session_and_run_RDM_plus_LOSO_group_RDM_consistency_not_test_retest',input_sha256=inputs,
      git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
      output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print('HCP RSA complete',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
