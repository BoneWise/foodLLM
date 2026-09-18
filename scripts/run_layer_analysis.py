"""Descriptive out-of-food-group estimates; deliberately no p-values or CIs.

Inference is a separate full-refit resampling stage. Stored errors must not be
treated as independent trials or subjects for naive significance testing.
"""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from batched_readout import crossfit_predict, tune_predict
from feature_inputs import load_layer_inputs
from two_stage_estimator import fit_ols_predict, leave_subject_out_mean


def brain_from_q(q, beta, content, probs, train, test, runs, sessions, held_session=None):
    """Same low-dimensional fit as neural_fold; reuse outcome-isolated q."""
    a = train[np.isfinite(beta[train])]
    b = test[np.isfinite(beta[test])]
    if held_session is not None:
        a = a[sessions[a] != held_session]
        b = b[sessions[b] == held_session]
    if not len(a) or not len(b):
        raise ValueError('Empty brain training or test fold')
    base = np.column_stack([content, probs[:, :3]])
    if held_session is None:
        levels = sorted(set(runs[a]))
        assert set(runs[b]).issubset(levels)
        if len(levels)>1:
            base = np.column_stack([base,*[(runs==r).astype(float) for r in levels[1:]]])
    predictions = {}
    ranks = {}
    for c in ['baseline',*q]:
        x = base if c=='baseline' else np.column_stack([base,q[c]])
        pred, model = fit_ols_predict(x[a],beta[a],x[b])
        predictions[c] = pred
        ranks[c] = [model['rank'],x.shape[1]+1]
    return b,predictions,ranks


def analyze_layer(data,lambdas,retain_q=False):
    features=data['features'];conditions=list(features)
    human=data['human_ratings'];beta=data['roi_beta'];s,n=human.shape
    h=np.nanmean(human,axis=0)
    targets=np.column_stack([leave_subject_out_mean(human,j) for j in range(s)])
    groups=data['food_groups'];folds=data['outer_folds']
    pred_h=np.full((len(conditions),n),np.nan)
    pred_b=np.full((2,s,len(conditions)+1,n),np.nan)
    test_counts=np.zeros((2,s,n),dtype=np.int8)
    traces=[];q_saved={};rank_deficient=0
    for fold in sorted(set(folds)):
        train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
        bh,chosen,losses=tune_predict(features,h,groups,train,test,lambdas,4)
        for k,c in enumerate(conditions):
            pred_h[k,test]=bh[c][:,0]
        q,qtraces=crossfit_predict(features,targets,groups,train,test,lambdas)
        traces.append({'outer_fold':int(fold),'behavior_lambda':float(chosen[0]),
                       'behavior_inner_losses':losses[:,0].tolist(),'neural_fits':qtraces})
        if retain_q:
            for c in conditions:
                q_saved[f'q_fold{fold}_{c}']=q[c]
        for subject in range(s):
            qs={c:q[c][:,subject] for c in conditions}
            for mode in range(2):
                sessions=([None] if mode==0 else
                          sorted(set(data['session_ids'][subject,np.isfinite(beta[subject])])))
                for held_session in sessions:
                    b,predictions,ranks=brain_from_q(qs,beta[subject],data['content_covariates'],
                        data['raw_V_probs'],train,test,data['run_ids'][subject],
                        data['session_ids'][subject],held_session)
                    assert not np.any(test_counts[mode,subject,b])
                    test_counts[mode,subject,b]+=1
                    for k,c in enumerate(['baseline',*conditions]):
                        pred_b[mode,subject,k,b]=predictions[c]
                        rank_deficient += int(ranks[c][0]<ranks[c][1])
    observed=np.isfinite(beta)
    assert np.array_equal(test_counts,np.broadcast_to(observed,(2,s,n)).astype(np.int8))
    assert np.isfinite(pred_h).all()
    assert np.array_equal(np.isfinite(pred_b),np.broadcast_to(observed[:,None,:],pred_b.shape))
    errors_h=(pred_h-h)**2
    errors_b=(pred_b-beta[None,:,None,:])**2
    # Pool observed held-out trials within each person, then weight people equally.
    mse_h=errors_h.mean(1)
    subject_mse_b=np.nanmean(errors_b,axis=-1)
    mse_b=subject_mse_b.mean(1)
    metrics={'behavior':dict(zip(conditions,mse_h.tolist()))}
    for k,mode in enumerate(['food','food_session']):
        metrics[mode]=dict(zip(['baseline',*conditions],mse_b[k].tolist()))
    for mode in metrics.values():
        mode['G']=mode['V']-mode['T']
        if len(conditions)>2:
            mode['P_mean']=float(np.mean([mode[c] for c in conditions if c.startswith('P')]))
            mode['S']=mode['P_mean']-mode['T']
        if 'baseline' in mode:
            mode['T_vs_baseline']=mode['baseline']-mode['T']
    return {'behavior_predictions':pred_h,'brain_predictions':pred_b,
            'behavior_squared_errors':errors_h,'brain_squared_errors':errors_b,
            'brain_subject_mse':subject_mse_b,'behavior_target':h,
            'test_counts':test_counts,**q_saved}, {
        'metrics':metrics,'conditions':conditions,'brain_conditions':['baseline',*conditions],
        'brain_modes':['food','food_session'],'fold_traces':traces,
        'rank_deficient_brain_fits':rank_deficient,'image_ids':data['image_ids'],
        'subject_ids':data['subject_ids'],'observed_trials':int(observed.sum())}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    args=ap.parse_args();config_path=Path(args.config)
    config=json.loads(config_path.read_text());out=Path(config['output'])
    out.mkdir(parents=True,exist_ok=False)
    scripts=['batched_readout.py','run_layer_analysis.py','feature_inputs.py','two_stage_estimator.py']
    snapshots=out/'scripts';snapshots.mkdir()
    for name in scripts:
        shutil.copy2(Path('scripts')/name,snapshots/name)
    shutil.copy2(config_path,out/'execution_config.json')
    inputs=[Path(config['matrix_root'])/'audit.json',Path(config['analysis_root'])/'audit.json',
            Path(config['matrix_root'])/'loader_validation.json',Path(config['validation'])]
    inputs += sorted(Path(config['analysis_root']).glob('*.npy'))
    inputs += [Path(config['analysis_root'])/'index.json']
    audit={'status':'running','scope':'descriptive_crossvalidated_effects_not_confirmatory_inference',
           'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'config_sha256':sha(config_path),'input_sha256':{str(p):sha(p) for p in inputs},
           'script_sha256':{name:sha(Path('scripts')/name) for name in scripts},
           'dependencies':{'python':platform.python_version(),'numpy':np.__version__,
                           'scipy':scipy.__version__,'sklearn':sklearn.__version__},
           'bootstrap_performed':False,'completed':[]}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    summary=[]
    with threadpool_limits(limits=config['blas_threads']):
        for sample in config['samples']:
            for prompt in config['prompts']:
                for layer in config['layers']:
                    start=time.monotonic();name=f'{sample}_{prompt}_layer{layer:02d}'
                    data=load_layer_inputs(config['matrix_root'],config['analysis_root'],prompt,layer,sample)
                    arrays,report=analyze_layer(data,config['lambdas'],config['retain_crossfit_q'])
                    np.savez_compressed(out/(name+'.npz'),**arrays)
                    report.update({'sample':sample,'prompt':prompt,'layer':layer,
                                   'elapsed_seconds':time.monotonic()-start})
                    (out/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
                    for endpoint,metrics in report['metrics'].items():
                        summary.append({'sample':sample,'prompt':prompt,'layer':layer,
                                        'endpoint':endpoint,**metrics})
                    audit['completed'].append(name)
                    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
                    print(f'{name}: completed in {time.monotonic()-start:.1f}s',flush=True)
    keys=['sample','prompt','layer','endpoint','baseline','V','T','P1','P2','P3','P4','P5',
          'P_mean','G','S','T_vs_baseline']
    with (out/'layer_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(summary)
    audit['status']='complete'
    audit['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    audit['output_sha256']={p.name:sha(p) for p in sorted(out.iterdir())
                          if p.suffix in ('.npz','.json','.csv') and p.name!='audit.json'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    main()
