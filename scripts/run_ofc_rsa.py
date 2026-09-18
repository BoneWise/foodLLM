"""Reuse frozen kcal model RDMs for raw lOFC/mOFC descriptive RSA."""
import argparse,json,shutil,platform,subprocess
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
CONDS=['V','T','P1','P2','P3','P4','P5']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--patterns',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    patterns=ROOT/args.patterns;out=(ROOT/args.output).resolve();assert out.is_relative_to(ROOT/'outputs') and not out.exists()
    pa=json.loads((patterns/'audit.json').read_text());assert pa['status']=='complete'
    old=ROOT/'outputs/20260917_raw_RSA_v01';oa=json.loads((old/'audit.json').read_text());assert oa['status']=='complete'
    idx=json.loads((old/'image_index.json').read_text());ids=idx['image_ids'];subjects=idx['subject_ids'];assert len(ids)==493 and len(subjects)==29
    out.mkdir(parents=True);shutil.copy2(__file__,out/'script_snapshot.py');shutil.copy2(ROOT/'scripts/rsa_core.py',out/'rsa_core_snapshot.py')
    (out/'image_index.json').write_text(json.dumps(idx,indent=2))
    inputs={str(p):sha(p) for p in [patterns/'audit.json',old/'audit.json',old/'image_index.json',Path(__file__),ROOT/'scripts/rsa_core.py']}
    check=validate();brains=[];qc=[];masks={}
    kcal=np.load(old/'kcal_RDM.npy');want=np.load(old/'wanting_RDM.npy')
    for roi in ['lOFC','mOFC']:
        ix=json.loads((patterns/roi/'index.json').read_text());assert ix['image_ids']==ids and ix['subject_ids']==subjects
        for sub in subjects:
            p=patterns/roi/sub/'patterns.npy';assert sha(p)==pa['output_sha256'][str(p.relative_to(patterns))]
            tt=rows(patterns/roi/sub/'trials.csv');ii=np.array([ids.index(r['image_id']) for r in tt]);assert len(ii)==len(set(ii))
            old_ii=np.load(old/'brain'/sub/'global_image_rows.npy');assert np.array_equal(ii,old_ii)
            d=correlation_rdm(np.load(p));tri=np.tril_indices(len(ii),-1);v=d[tri];rank=ranked_unit(v)
            sd=out/roi/'brain'/sub;sd.mkdir(parents=True);np.save(sd/'RDM.npy',d);np.save(sd/'global_image_rows.npy',ii)
            gr,gc=ii[tri[0]],ii[tri[1]];mk=tuple(ii);masks[mk]=(gr,gc)
            sess=np.array([r['session_id'] for r in tt]);run=np.array([r['run_identity'] for r in tt])
            rec={'ROI':roi,'subject_id':sub,'images':len(ii),'pairs':len(v),'voxels':np.load(p,mmap_mode='r').shape[1],
                 'kcal_rho':score(kcal[gr,gc],v),'wanting_rho':score(want[gr,gc],v)}
            for name,lab in [('session',sess),('run',run)]:
                x=lab[tri[0]]!=lab[tri[1]];rec[name+'_rho']=score(x,v)
                rec[name+'_same_mean_distance']=float(v[~x].mean());rec[name+'_different_mean_distance']=float(v[x].mean())
            qc.append(rec);brains.append({'ROI':roi,'subject':sub,'maskkey':mk,'ranks':rank,'vector':v})
    write_csv(out/'acquisition_QC.csv',qc);print('58 brain RDMs complete; exact historical trial masks retained',flush=True)
    neural=[];contrasts=[];curves=[];validation={};all_scores={}
    for prompt in ['A','B']:
        if prompt=='B':assert validation['A']['passed']
        result=np.empty((2,29,32,7));errors=[]
        for ci,c in enumerate(CONDS):
            p=old/prompt/(c+'_RDM.npy');digest=sha(p);assert digest==oa['output_sha256'][str(p.relative_to(old))];inputs[str(p)]=digest
            model=np.load(p,mmap_mode='r');assert model.shape==(32,493,493)
            for li in range(32):
                ranks={mk:ranked_unit(model[li,rr,cc]) for mk,(rr,cc) in masks.items()}
                for bi,b in enumerate(brains):
                    ri=bi//29;si=bi%29;rho=float(ranks[b['maskkey']]@b['ranks']);result[ri,si,li,ci]=rho
                    if li in [0,15,31] and si in [0,15,28]:
                        rr,cc=masks[b['maskkey']];errors.append(abs(rho-spearmanr(model[li,rr,cc],b['vector']).statistic))
            print(f'{prompt} {c}: all layers and both ROIs',flush=True)
        assert np.isfinite(result).all() and max(errors)<1e-12
        for ri,roi in enumerate(['lOFC','mOFC']):
            scores=result[ri];pmean=scores[:,:,2:].mean(2);delta=np.stack([scores[:,:,1]-scores[:,:,0],scores[:,:,1]-pmean],axis=2)
            pd=out/roi/prompt;pd.mkdir();np.save(pd/'neural_scores.npy',scores);np.save(pd/'contrasts.npy',delta)
            for si,sub in enumerate(subjects):
                for li in range(32):
                    for ci,c in enumerate(CONDS):neural.append({'ROI':roi,'prompt':prompt,'subject_id':sub,'layer':li+1,'condition':c,'rho':float(scores[si,li,ci])})
                    contrasts.append({'ROI':roi,'prompt':prompt,'subject_id':sub,'layer':li+1,'T_minus_V':float(delta[si,li,0]),'T_minus_mean_P':float(delta[si,li,1])})
            for li in range(32):
                m=scores[:,li].mean(0)
                curves.append({'ROI':roi,'prompt':prompt,'layer':li+1,'V':float(m[0]),'T':float(m[1]),'P_mean':float(m[2:].mean()),
                  'P_min':float(m[2:].min()),'P_max':float(m[2:].max()),'T_minus_V':float(delta[:,li,0].mean()),'T_minus_mean_P':float(delta[:,li,1].mean())})
        validation[prompt]={'passed':True,'direct_scipy_checks':len(errors),'max_difference':max(errors),'B_gate':'correctness_only'}
        all_scores[prompt]=result
    for name,rr in [('neural_scores',neural),('contrasts',contrasts),('group_curves',curves)]:write_csv(out/(name+'.csv'),rr)
    summary={}
    for ri,roi in enumerate(['lOFC','mOFC']):
        summary[roi]={}
        for prompt,result in all_scores.items():
            s=result[ri];d=np.stack([s[:,:,1]-s[:,:,0],s[:,:,1]-s[:,:,2:].mean(2)],2)
            summary[roi][prompt]={'cross_layer_condition_means':dict(zip(CONDS,s.mean((0,1)).tolist())),
               'cross_layer_T_minus_V':float(d[:,:,0].mean()),'cross_layer_T_minus_P':float(d[:,:,1].mean()),
               'positive_group_layers_TV':int((d[:,:,0].mean(0)>0).sum()),'positive_group_layers_TP':int((d[:,:,1].mean(0)>0).sum())}
        qq=[r for r in qc if r['ROI']==roi]
        summary[roi]['QC_group_means']={k:float(np.mean([r[k] for r in qq])) for k in ['kcal_rho','wanting_rho','session_rho','run_rho']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    audit={'status':'complete','created':datetime.now().astimezone().isoformat(),'new_model_inference':False,
      'brain_RDMs':58,'reused_model_RDMs':448,'neural_scores':len(neural),'new_exclusions':[],
      'extra_brain_preprocessing':[],'inference_run':False,'sensitivity_run':False,'validation':validation,'numerical_fixtures':check,
      'input_sha256':inputs,'seed':None,'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
      'output_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
