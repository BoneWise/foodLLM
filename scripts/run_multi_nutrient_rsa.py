"""One prompt of four frozen nutrient experiments on a common image set and three ROIs."""
import argparse,csv,json,shutil,platform,subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from prepare_roi import sha,write_csv
from extract_rsa_patterns import rows
from rsa_core import correlation_rdm,ranked_unit,score,validate

ROOT=Path(__file__).resolve().parents[1]
ROIS=['lOFC','mOFC','vmPFC'];CONDS=['V','T','P1','P2','P3','P4','P5']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prompt',choices=['A','B'],required=True)
    for name in ['matrices','output']:ap.add_argument('--'+name,required=True)
    args=ap.parse_args();prompt=args.prompt;matrixroot=ROOT/args.matrices;out=(ROOT/args.output).resolve()
    assert out.is_relative_to(ROOT/'outputs') and not out.exists()
    cfgp=ROOT/'configs/multi_nutrient_execution_v01.json';cfg=json.loads(cfgp.read_text());prep=ROOT/cfg['metadata_preparation']
    assert sha(prep/'stimulus_manifest.csv')==cfg['manifest_sha256']['stimulus_manifest.csv']
    ma=json.loads((matrixroot/'audit.json').read_text());assert ma['status']=='complete_verified' and ma['verified_inputs']==9624
    ids=[r['image_id'] for r in rows(prep/'stimulus_manifest.csv')];assert ma['common_image_ids']==ids and len(ids)==401
    nutrients=cfg['nutrients'];old=ROOT/'outputs/20260917_raw_RSA_v01';ofc=ROOT/'outputs/20260917_OFC_kcal_RSA_v01'
    olda=json.loads((old/'audit.json').read_text());ofca=json.loads((ofc/'audit.json').read_text())
    assert olda['status']==ofca['status']=='complete'
    ix=json.loads((old/'image_index.json').read_text());subjects=ix['subject_ids'];oldids=ix['image_ids'];take=np.array([oldids.index(i) for i in ids]);index={i:n for n,i in enumerate(ids)}
    n=len(ids);tri=np.tril_indices(n,-1);out.mkdir(parents=True);shutil.copy2(__file__,out/'script_snapshot.py');shutil.copy2(cfgp,out/'execution_snapshot.json')
    inp={str(p):sha(p) for p in [cfgp,prep/'stimulus_manifest.csv',matrixroot/'audit.json',old/'audit.json',ofc/'audit.json',Path(__file__),ROOT/'scripts/rsa_core.py']}
    numerical=validate();brains=[];masks={};qc=[];references=[]
    for p in [old/'wanting_RDM.npy',old/'image_index.json']:
        assert sha(p)==olda['output_sha256'][str(p.relative_to(old))];inp[str(p)]=sha(p)
    wanting=np.load(old/'wanting_RDM.npy')[np.ix_(take,take)];wrank=ranked_unit(wanting[tri]);np.save(out/'wanting_RDM.npy',wanting)
    metadata={r['image_id']:r for r in rows(prep/'stimulus_manifest.csv')}
    nr={}
    for nutrient in nutrients:
        values=np.array([float(metadata[i][nutrient]) for i in ids]);nr[nutrient]=np.abs(values[:,None]-values[None,:]);np.save(out/(nutrient+'_reference_RDM.npy'),nr[nutrient])
    for roi in ROIS:
        source=old if roi=='vmPFC' else ofc/roi;sa=olda if roi=='vmPFC' else ofca;base=old if roi=='vmPFC' else ofc
        for sub in subjects:
            path=source/'brain'/sub/'RDM.npy';ip=source/'brain'/sub/'global_image_rows.npy'
            for p in [path,ip]:assert sha(p)==sa['output_sha256'][str(p.relative_to(base))];inp[str(p)]=sha(p)
            oldrows=np.load(ip);si=np.array([j for j,k in enumerate(oldrows) if oldids[k] in index]);globalrows=np.array([index[oldids[oldrows[j]]] for j in si]);assert np.all(np.diff(globalrows)>0)
            d=np.load(path)[np.ix_(si,si)];lower=np.tril_indices(len(si),-1);v=d[lower];rank=ranked_unit(v);gr,gc=globalrows[lower[0]],globalrows[lower[1]];mk=tuple(globalrows)
            masks[mk]=(gr,gc);sd=out/roi/'brain'/sub;sd.mkdir(parents=True);np.save(sd/'RDM.npy',d);np.save(sd/'global_image_rows.npy',globalrows)
            ds=np.load(old/'brain'/sub/'session_RDM.npy')[np.ix_(si,si)];dr=np.load(old/'brain'/sub/'run_RDM.npy')[np.ix_(si,si)]
            assert np.array_equal(np.load(old/'brain'/sub/'global_image_rows.npy'),oldrows)
            qc.append({'ROI':roi,'subject_id':sub,'images':len(si),'pairs':len(v),'session_rho':score(ds[lower],v),'run_rho':score(dr[lower],v),'wanting_rho':score(wanting[gr,gc],v)})
            for nutrient in nutrients:references.append({'ROI':roi,'subject_id':sub,'nutrient':nutrient,'nutrient_brain_rho':score(nr[nutrient][gr,gc],v),'nutrient_wanting_rho':score(nr[nutrient][tri],wanting[tri])})
            brains.append({'ROI':roi,'subject':sub,'maskkey':mk,'ranks':rank,'vector':v})
    assert all(brains[s]['maskkey']==brains[s+29]['maskkey']==brains[s+58]['maskkey'] for s in range(29))
    write_csv(out/'acquisition_QC.csv',qc);write_csv(out/'nutrient_references.csv',references)
    expected={name:sha256 for g in ma['groups'] for name,sha256 in g['files'].items()}
    vpath=old/prompt/'V_RDM.npy';assert sha(vpath)==olda['output_sha256'][str(vpath.relative_to(old))];inp[str(vpath)]=sha(vpath)
    vrdm=np.load(vpath,mmap_mode='r');vscore=np.empty((87,32));vbehavior=np.empty(32);errors=[]
    def relatedness(d,li):
        ranks={mk:ranked_unit(d[rr,cc]) for mk,(rr,cc) in masks.items()}
        scores=np.array([float(ranks[b['maskkey']]@b['ranks']) for b in brains]);behavior=float(ranked_unit(d[tri])@wrank)
        if li in [0,15,31]:
            for bi in [0,15,28,29,44,57,58,73,86]:
                b=brains[bi];rr,cc=masks[b['maskkey']];errors.append(abs(scores[bi]-spearmanr(d[rr,cc],b['vector']).statistic))
            errors.append(abs(behavior-spearmanr(d[tri],wanting[tri]).statistic))
        return scores,behavior
    for li in range(32):vscore[:,li],vbehavior[li]=relatedness(vrdm[li][np.ix_(take,take)],li)
    np.save(out/'shared_V_neural_scores.npy',vscore.reshape(3,29,32));np.save(out/'shared_V_behavior_scores.npy',vbehavior)
    curves=[];neural=[];behaviorrows=[];contrastrows=[]
    for nutrient in nutrients:
        scores=np.empty((87,32,7));scores[:,:,0]=vscore;behavior=np.empty((32,7));behavior[:,0]=vbehavior
        nd=out/nutrient;nd.mkdir()
        for ci,c in enumerate(CONDS[1:],1):
            group=matrixroot/f'{prompt}_{nutrient}_{c}'
            for name in ['hidden.npy','image_index.csv']:
                p=group/name;digest=sha(p);assert digest==expected[str(p.relative_to(matrixroot))];inp[str(p)]=digest
            rr=rows(group/'image_index.csv');assert [r['image_id'] for r in rr]==ids
            features=np.load(group/'hidden.npy',mmap_mode='r');assert features.shape==(32,401,4096)
            rdms=np.lib.format.open_memmap(nd/(c+'_RDM.npy'),mode='w+',dtype='float64',shape=(32,n,n))
            for li in range(32):
                d=correlation_rdm(features[li]);rdms[li]=d;scores[:,li,ci],behavior[li,ci]=relatedness(d,li)
            rdms.flush();del rdms;print(f'{prompt} {nutrient} {c}: all layers / three ROIs',flush=True)
        scores=scores.reshape(3,29,32,7);delta=np.stack([scores[:,:,:,1]-scores[:,:,:,0],scores[:,:,:,1]-scores[:,:,:,2:].mean(3)],3)
        assert np.isfinite(scores).all() and np.isfinite(behavior).all()
        np.save(nd/'neural_scores.npy',scores);np.save(nd/'behavior_scores.npy',behavior);np.save(nd/'contrasts.npy',delta)
        for ri,roi in enumerate(ROIS):
            for si,sub in enumerate(subjects):
                for li in range(32):
                    for ci,c in enumerate(CONDS):neural.append({'ROI':roi,'prompt':prompt,'nutrient':nutrient,'subject_id':sub,'layer':li+1,'condition':c,'rho':float(scores[ri,si,li,ci])})
                    contrastrows.append({'ROI':roi,'prompt':prompt,'nutrient':nutrient,'subject_id':sub,'layer':li+1,'T_minus_V':float(delta[ri,si,li,0]),'T_minus_mean_P':float(delta[ri,si,li,1])})
            for li in range(32):
                m=scores[ri,:,li].mean(0)
                curves.append({'ROI':roi,'prompt':prompt,'nutrient':nutrient,'layer':li+1,'V':float(m[0]),'T':float(m[1]),'P_mean':float(m[2:].mean()),'P_min':float(m[2:].min()),'P_max':float(m[2:].max()),'T_minus_V':float(delta[ri,:,li,0].mean()),'T_minus_mean_P':float(delta[ri,:,li,1].mean())})
        for li in range(32):
            for ci,c in enumerate(CONDS):behaviorrows.append({'prompt':prompt,'nutrient':nutrient,'layer':li+1,'condition':c,'rho':float(behavior[li,ci])})
    assert max(errors)<1e-12
    for name,rr in [('group_curves',curves),('neural_scores',neural),('behavior_scores',behaviorrows),('contrasts',contrastrows)]:write_csv(out/(name+'.csv'),rr)
    index_out={'image_ids':ids,'subject_ids':subjects,'ROIs':ROIS,'nutrients':nutrients,'conditions':CONDS,'prompt':prompt,'layers':list(range(1,33))}
    (out/'index.json').write_text(json.dumps(index_out,indent=2))
    audit={'status':'complete','prompt':prompt,'images':401,'participants':29,'brain_RDMs':87,'new_model_RDMs':768,'reused_V_layers':32,
      'brain_trials_per_ROI':sum(q['images'] for q in qc if q['ROI']=='vmPFC'),'neural_scores':len(neural),
      'all_conditions_attributes_ROIs_same_pair_masks':True,'V_shared_exact':True,'extra_brain_processing':[],
      'inference_run':False,'sensitivity_run':False,'numerical_fixtures':numerical,'scipy_checks':len(errors),'max_scipy_difference':max(errors),
      'created':datetime.now().astimezone().isoformat(),'input_sha256':inp,'seed':None,
      'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
      'output_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print('Common-set multi-nutrient RSA complete',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
