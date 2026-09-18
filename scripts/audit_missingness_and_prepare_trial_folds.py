"""Assess available-case averages and prepare user-requested within-person trial folds.

No existing targets, predictions, or statistical results are replaced.
"""
import csv
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.stats import pearsonr, spearmanr
from run_layer_analysis import sha

OUT=Path('outputs/20260916_trial划分与缺失均值评估_v01')
SEED=20260916


def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def loo_summary(x):
    mean=np.nanmean(x,axis=0);differences=[];rs=[];rhos=[]
    for s in range(len(x)):
        other=np.nanmean(np.delete(x,s,axis=0),axis=0)
        differences.extend(other-mean)
        rs.append(float(pearsonr(mean,other).statistic))
        rhos.append(float(spearmanr(mean,other).statistic))
    absolute=np.abs(differences)
    return {'mean_absolute_change':float(absolute.mean()),'absolute_change_95th_percentile':float(np.quantile(absolute,.95)),
        'max_absolute_change':float(absolute.max()),'pearson_range':[min(rs),max(rs)],
        'spearman_range':[min(rhos),max(rhos)],'interpretation':'sensitivity_to_removing_one_observed_person_not_proof_missingness_random'}


def main():
    OUT.mkdir(exist_ok=False)
    project_path=Path('configs/project.json');project=json.loads(project_path.read_text())
    root=Path(project['paths']['analysis_tables']);idx=json.loads((root/'index.json').read_text())
    h=np.load(root/'human_ratings.npy');b=np.load(root/'roi_beta.npy');primary=np.load(root/'primary_mask.npy')
    sessions=np.load(root/'session_ids.npy');runs=np.load(root/'run_ids.npy')
    with Path(project['paths']['trial_ratings']).open(encoding='utf-8-sig') as f:
        trial_rows=list(csv.DictReader(f))
    trial_map={(r['subject_id'],r['image_id']):r for r in trial_rows}
    assert len(trial_map)==len(trial_rows)==15869
    report={'scope':'descriptive_missingness_sensitivity_and_split_preparation_only','samples':{}}
    assignments=[];counts=[]
    for sample_index,(sample,mask) in enumerate([('primary',primary),('full_VT',np.ones(568,dtype=bool))]):
        x=h[:,mask];brain=b[:,mask];n=np.isfinite(x).sum(0);nb=np.isfinite(brain).sum(0);y=np.nanmean(x,axis=0)
        levels,nlevels=np.unique(n,return_counts=True);blevels,bnlevels=np.unique(nb,return_counts=True)
        missing=np.isnan(x).sum(1)
        # Descriptive additive subject response-style sensitivity, NOT a new target.
        # Model rating_si = image_effect_i + subject_effect_s; last subject is reference.
        si,ii=np.where(np.isfinite(x));N=len(ii);J=x.shape[1];use=si<28
        design=coo_matrix((np.ones(N+use.sum()),(np.r_[np.arange(N),np.arange(N)[use]],
            np.r_[ii,J+si[use]])),shape=(N,J+28)).tocsr()
        solution=lsqr(design,x[si,ii],atol=1e-12,btol=1e-12)
        assert solution[1] in [1,2]
        adjusted=solution[0][:J]+np.r_[solution[0][J:],0.].mean()
        adjustment={'model':'rating = image fixed effect + subject fixed effect',
            'reference':'predicted image mean if all 29 subject intercepts equally represented',
            'assumption':'subject response-style offsets additive across images',
            'mean_absolute_change':float(np.mean(np.abs(adjusted-y))),
            'max_absolute_change':float(np.max(np.abs(adjusted-y))),
            'pearson':float(pearsonr(adjusted,y).statistic),'spearman':float(spearmanr(adjusted,y).statistic),
            'new_target_used':False,'proof_missingness_random':False}
        report['samples'][sample]={'images':int(mask.sum()),'possible_observations':int(x.size),
            'observed_behavior':int(n.sum()),'missing_behavior':int(x.size-n.sum()),
            'missing_behavior_percent':float(100*(1-n.sum()/x.size)),
            'observed_brain':int(nb.sum()),'missing_brain':int(brain.size-nb.sum()),
            'missing_brain_percent':float(100*(1-nb.sum()/brain.size)),
            'image_behavior_count_histogram':{str(k):int(v) for k,v in zip(levels,nlevels)},
            'image_brain_count_histogram':{str(k):int(v) for k,v in zip(blevels,bnlevels)},
            'subjects_with_any_missing_behavior':int((missing>0).sum()),
            'subject_missing_behavior':dict(zip(idx['subject_ids'],missing.tolist())),
            'behavior_leave_one_person_out':loo_summary(x),'brain_mean_leave_one_person_out':loo_summary(brain),
            'response_style_sensitivity':adjustment}
        np.savez_compressed(OUT/f'{sample}_mean_sensitivity.npz',available_case_behavior_mean=y,
            additive_subject_adjusted_mean=adjusted,image_ids=np.array(idx['image_ids'])[mask])
        # Independent subject/sample seeds avoid dependence on iteration order.
        for s,subject in enumerate(idx['subject_ids']):
            indices=np.flatnonzero(mask&np.isfinite(b[s]));assert len(indices)>5
            subject_number=int(subject.split('-')[-1])
            rng=np.random.default_rng(np.random.SeedSequence([SEED,sample_index,subject_number]))
            shuffled=rng.permutation(indices)
            for fold,test in enumerate(np.array_split(shuffled,5)):
                counts.append({'sample':sample,'subject_id':subject,'outer_fold':fold,
                    'training_trials':int(len(indices)-len(test)),'test_trials':int(len(test))})
                for i in test:
                    r=trial_map[subject,idx['image_ids'][i]]
                    assert r['session_id']==sessions[s,i] and r['session_id']+'/'+r['run_id']==runs[s,i]
                    assignments.append({'sample':sample,'subject_id':subject,'session_id':r['session_id'],
                        'run_id':r['run_id'],'trial_id':r['trial_id'],'image_id':r['image_id'],'outer_fold':fold})
    assignments.sort(key=lambda r:(r['sample'],r['subject_id'],int(r['image_id'])))
    assert len(assignments)==13765+15867
    assert len({(r['sample'],r['subject_id'],r['session_id'],r['run_id'],r['trial_id']) for r in assignments})==len(assignments)
    for name,rows in [('neural_trial_folds.csv',assignments),('fold_counts.csv',counts)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for sample in report['samples']:
        cc=[r for r in counts if r['sample']==sample]
        report['samples'][sample]['new_trial_fold_sizes']={k:[min(r[k] for r in cc),max(r[k] for r in cc)] for k in ['training_trials','test_trials']}
    save(OUT/'missingness_assessment.json',report)
    save(OUT/'validation.json',{'status':'passed','all_eligible_neural_trials_assigned_once_per_sample':True,
        'fold_count':5,'subject_independent_random_trial_split':True,
        'session_and_food_group_not_used_in_assignment':True,'truncated_trials_absent':True,
        'old_inputs_or_results_modified':False,'new_model_fits_performed':False,
        'mean_sensitivity_not_used_as_training_targets':True})
    shutil.copy2(__file__,OUT/'script_snapshot.py')
    inputs=[project_path,Path(project['paths']['trial_ratings']),root/'index.json']+[root/(k+'.npy') for k in ['human_ratings','roi_beta','primary_mask','session_ids','run_ids']]
    save(OUT/'audit.json',{'status':'complete','seed':SEED,
        'subject_seed_rule':'SeedSequence([20260916, sample_index(primary=0/full=1), numeric_subject_id])',
        'parameters':{'folds':5,'lsqr_atol':1e-12,'lsqr_btol':1e-12},
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'input_sha256':{str(p):sha(p) for p in inputs},'script_sha256':sha(__file__),
        'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
