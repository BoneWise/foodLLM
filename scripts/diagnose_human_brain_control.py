"""Descriptive human-rating/valuation-beta controls; no inferential testing.

Uses observed held-out human ratings as explanatory covariates. This is not
image-only prediction and is not a replacement for the model comparison.
"""
import argparse
import csv
import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from two_stage_estimator import fit_ols_predict,leave_subject_out_mean
from prepare_roi import sha,write_csv


def probability_matrix(root,prompt,ids):
    d=Path(root)/f'{prompt}_V'
    with (d/'image_index.csv').open() as f:
        mapping={r['image_id']:int(r['matrix_row']) for r in csv.DictReader(f)}
    return np.load(d/'rating_probs.npy')[[mapping[i] for i in ids]]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--timing-flags');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    root=Path('outputs/20260916_分析输入索引_v01');idx=json.loads((root/'index.json').read_text())
    a={k:np.load(root/(k+'.npy')) for k in ['human_ratings','roi_beta','session_ids','run_ids',
       'content_covariates','outer_folds','primary_mask']}
    config={'scope':'post-estimate descriptive positive-control diagnostics, no p-values or CIs',
        'samples':['full_VT','primary'],'modes':['food','food_session'],
        'baselines':['run_or_global_intercept','full_A','full_B'],
        'added_covariates':['own_observed_rating','other_subject_mean_observed_rating'],
        'fitting':'OLS; training-only standardization; existing five food folds',
        'availability_note':'Test human ratings are deliberately observed explanatory inputs, not predicted inputs. Other-mean excludes target identity.',
        'no_model_or_ROI_selection':True,'seed':None}
    if args.timing_flags:
        with Path(args.timing_flags).open() as f:flags=list(csv.DictReader(f))
        for r in flags:
            si=idx['subject_ids'].index(r['subject_id']);ii=idx['image_ids'].index(r['image_id'])
            assert a['session_ids'][si,ii]==r['session_id']
            assert a['run_ids'][si,ii]==r['session_id']+'/'+r['run_id']
            a['roi_beta'][si,ii]=np.nan
        config['timing_sensitivity']={'file':args.timing_flags,'sha256':sha(args.timing_flags),
            'brain_observations_omitted':len(flags),'human_ratings_unchanged':True,
            'does_not_replace_original_analysis':True}
    (out/'diagnostic_config.json').write_text(json.dumps(config,indent=2)+'\n')
    records=[];correlations=[]
    with threadpool_limits(limits=2):
        for sample in config['samples']:
            ids=np.flatnonzero(a['primary_mask']) if sample=='primary' else np.arange(len(idx['image_ids']))
            image_ids=[idx['image_ids'][i] for i in ids]
            h=a['human_ratings'][:,ids];beta=a['roi_beta'][:,ids];folds=a['outer_folds'][ids]
            sessions=a['session_ids'][:,ids];runs=a['run_ids'][:,ids];content=a['content_covariates'][ids]
            probs={p:probability_matrix('outputs/20260916_Qwen_AB分析矩阵_v01',p,image_ids) for p in ['A','B']}
            for si,subject in enumerate(idx['subject_ids']):
                observed=np.isfinite(beta[si]);own=h[si];other=leave_subject_out_mean(h,si)
                assert np.isfinite(own[observed]).all()
                # Descriptive correlation after removing each run's observed mean.
                centered=[]
                for rr in sorted(set(runs[si,observed])):
                    mask=observed&(runs[si]==rr)
                    centered.append(np.column_stack([own[mask]-own[mask].mean(),
                        other[mask]-other[mask].mean(),beta[si,mask]-beta[si,mask].mean()]))
                centered=np.concatenate(centered)
                correlations.append({'sample':sample,'subject_id':subject,'trials':int(observed.sum()),
                    'within_run_own_rating_beta_r':float(np.corrcoef(centered[:,0],centered[:,2])[0,1]),
                    'within_run_other_mean_beta_r':float(np.corrcoef(centered[:,1],centered[:,2])[0,1])})
                for mode in config['modes']:
                    for baseline in config['baselines']:
                        predictions=np.full((3,len(ids)),np.nan);counts=np.zeros(len(ids),int);slopes=[[],[]]
                        for fold in sorted(set(folds)):
                            train=np.flatnonzero((folds!=fold)&observed);test=np.flatnonzero((folds==fold)&observed)
                            helds=[None] if mode=='food' else sorted(set(sessions[si,observed]))
                            for held in helds:
                                tr=train if held is None else train[sessions[si,train]!=held]
                                te=test if held is None else test[sessions[si,test]==held]
                                assert len(tr)>0 and len(te)>0
                                base=np.empty((len(ids),0)) if baseline=='run_or_global_intercept' else np.column_stack([content,probs[baseline[-1]][:,:3]])
                                if held is None:
                                    levels=sorted(set(runs[si,tr]));assert set(runs[si,te]).issubset(levels)
                                    if len(levels)>1:base=np.column_stack([base,*[(runs[si]==r).astype(float) for r in levels[1:]]])
                                inputs=[base,np.column_stack([base,own]),np.column_stack([base,other])]
                                for k,x in enumerate(inputs):
                                    p,model=fit_ols_predict(x[tr],beta[si,tr],x[te]);predictions[k,te]=p
                                    if k:slopes[k-1].append(float(model['weight'][-1]/model['scale'][-1]))
                                counts[te]+=1
                        assert np.array_equal(counts,observed.astype(int))
                        mse=np.nanmean((predictions-beta[si])**2,axis=1)
                        records.append({'sample':sample,'subject_id':subject,'mode':mode,'baseline':baseline,
                            'observed_trials':int(observed.sum()),'baseline_MSE':float(mse[0]),
                            'own_rating_MSE':float(mse[1]),'other_mean_MSE':float(mse[2]),
                            'own_gain':float(mse[0]-mse[1]),'other_gain':float(mse[0]-mse[2]),
                            'own_mean_training_slope':float(np.mean(slopes[0])),
                            'other_mean_training_slope':float(np.mean(slopes[1]))})
                        np.savez_compressed(out/f'{sample}_{subject}_{mode}_{baseline}.npz',
                            predictions=predictions,observed=observed)
    summary=[]
    for sample in config['samples']:
        for mode in config['modes']:
            for base in config['baselines']:
                rr=[r for r in records if (r['sample'],r['mode'],r['baseline'])==(sample,mode,base)]
                assert len(rr)==29
                m={key:float(np.mean([r[key] for r in rr])) for key in ['baseline_MSE','own_rating_MSE','other_mean_MSE','own_gain','other_gain']}
                m.update({'sample':sample,'mode':mode,'baseline':base,
                    'own_gain_percent':100*m['own_gain']/m['baseline_MSE'],'other_gain_percent':100*m['other_gain']/m['baseline_MSE'],
                    'subjects_own_gain_positive':sum(r['own_gain']>0 for r in rr),
                    'subjects_other_gain_positive':sum(r['other_gain']>0 for r in rr)})
                summary.append(m)
    write_csv(out/'subject_cv_controls.csv',records);write_csv(out/'subject_within_run_correlations.csv',correlations)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copy2(__file__,out/'script_snapshot.py')
    audit={'status':'complete','created_at':datetime.now().astimezone().isoformat(),
        'config':config,'subjects':29,'subject_cv_records':len(records),'each_observed_trial_predicted_once':True,
        'hypothesis_inference_performed':False,'script_sha256':sha(__file__),
        'input_sha256':{p.name:sha(p) for p in root.glob('*.npy')},
        'V_probability_sha256':{p:sha(Path('outputs/20260916_Qwen_AB分析矩阵_v01')/f'{p}_V'/'rating_probs.npy') for p in ['A','B']},
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dependencies':{'python':platform.python_version(),'numpy':np.__version__},
        'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
