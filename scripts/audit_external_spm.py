"""Audit actual SPM timings, estimability and re-extracted ROI values; no GLM refit."""
import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import nibabel as nib
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
from prepare_roi import roi_values,sha,write_csv


def rows(path,delimiter=','):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f,delimiter=delimiter))


def listify(x):
    return [x] if isinstance(x,dict) else np.atleast_1d(x).tolist()


def stats(x):
    x=np.asarray(x,dtype=float)
    return {'min':float(x.min()),'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'max':float(x.max())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    mapping_path=Path('data/external/behavior_to_singletrial_beta_29subjects.csv');mapping=rows(mapping_path)
    roi_root=Path('outputs/20260916_vmPFC准备_v01');previous=json.loads((roi_root/'audit.json').read_text())
    old_inputs={r['spm_path']:r for r in previous['provenance']['run_input_hashes']}
    old_means={tuple(r[k] for k in ['subject_id','session_id','run_id','trial_id']):float(r['roi_beta_mean'])
               for r in rows(roi_root/'roi_trial_responses.csv')}
    manifest={r['image_id']:int(r['outer_fold']) for r in rows('outputs/20260916_刺激与错配准备_v03/stimulus_manifest.csv')}
    grouped=defaultdict(list)
    for r in mapping:grouped[(r['subject_id'],r['session_id'],r['run_id'])].append(r)
    ref=nib.load(mapping[0]['beta_file_local']);sphere=nib.load(roi_root/'vmPFC_sphere.nii.gz').get_fdata()>0
    vox=np.where(sphere);slices=tuple(slice(int(v.min()),int(v.max())+1) for v in vox)
    local=sphere[slices]
    supports={s:(nib.load(roi_root/f'{s}_fixed_support.nii.gz').get_fdata()>0)[vox] for s,_,_ in grouped}
    run_results=[];trial_results=[];errors=[];external_hashes=[];bf_configs=Counter();duration_errors=[]
    with threadpool_limits(limits=2):
        for j,(run,rr) in enumerate(sorted(grouped.items())):
            spm_path=Path(rr[0]['spm_mat_local']);spm=loadmat(spm_path,simplify_cells=True)['SPM']
            names=list(np.atleast_1d(spm['xX']['name']));u_map={str(np.atleast_1d(u['name'])[0]):u for u in listify(spm['Sess']['U'])}
            evpath=Path('data/raw/events')/run[0]/run[1]/'func'/f'{run[0]}_{run[1]}_task-food_run-{run[2]}_events.tsv'
            events=rows(evpath,'\t');ev={(e['trial_id'],e['trial_type']):e for e in events}
            bf=spm['xBF'];basis=np.asarray(bf['bf']);n_basis=1 if basis.ndim==1 else basis.shape[1]
            config=(str(bf['name']),n_basis,str(bf['UNITS']),float(spm['xY']['RT']),int(bf['T']),int(bf['T0']),float(spm['xX']['K']['HParam']),str(spm['xVi']['form']))
            bf_configs[str(config)]+=1
            assert config[:7]==('hrf',1,'secs',.8,60,30,128.)
            target_columns=[]
            for r in rr:
                target=f'target_trial_{int(r["trial_id"]):04d}';u=u_map[target];e=ev[(r['trial_id'],'valuation')]
                col=int(re.search(r'beta_(\d+)',Path(r['beta_file_local']).name)[1])-1
                assert names[col]==f'Sn(1) {target}*bf(1)'
                assert Path(spm['Vbeta'][col]['fname']).name==Path(r['beta_file_local']).name
                assert len(np.atleast_1d(u['ons']))==len(np.atleast_1d(u['dur']))==1
                onset_error=abs(float(np.atleast_1d(u['ons'])[0])-float(e['onset']))
                duration_error=abs(float(np.atleast_1d(u['dur'])[0])-float(e['duration']))
                assert onset_error<1e-6 and duration_error<1e-6
                assert u['P']['name']=='none' and int(u['P']['h'])==0
                duration_errors.append(duration_error);target_columns.append(col)
            for stage in ['rating','response','feedback']:
                ee=[e for e in events if e['trial_type']==stage];u=u_map[stage]
                assert np.allclose(np.atleast_1d(u['ons']),[float(e['onset']) for e in ee],rtol=0,atol=1e-6)
                dur=np.atleast_1d(u['dur'])
                if len(dur)==1:dur=np.repeat(dur,len(ee))
                assert np.allclose(dur,[float(e['duration']) for e in ee],rtol=0,atol=1e-6)
                assert u['P']['name']=='none'
            x=np.asarray(spm['xX']['xKXs']['X'],dtype=float)
            # Use SPM's own tolerance on the stored whitened/filtered design.
            _,sing,vt=svd(x,full_matrices=False,check_finite=False)
            rank=int(np.sum(sing>float(spm['xX']['xKXs']['tol'])))
            assert rank==int(spm['xX']['xKXs']['rk'])
            null_size=np.sqrt(np.sum(vt[rank:]**2,axis=0)) if rank<x.shape[1] else np.zeros(x.shape[1])
            p=np.asarray(spm['xX']['pKX'])
            row_projector=p@x
            target_estimability=np.linalg.norm((row_projector-np.eye(x.shape[1]))[:,target_columns],axis=0)
            bcov=np.asarray(spm['xX']['Bcov']);sd=np.sqrt(np.maximum(np.diag(bcov),1e-300))
            corr=bcov/np.outer(sd,sd)
            tc=np.asarray(target_columns);tcorr=corr[np.ix_(tc,tc)]
            folds=np.array([manifest[r['image_id']] for r in rr]);different=folds[:,None]!=folds[None,:]
            stage_cols=[names.index(f'Sn(1) {stage}*bf(1)') for stage in ['rating','response','feedback']]
            stage_corr=np.corrcoef(x[:,stage_cols].T)
            aggregate=x[:,tc].sum(1)
            val_stage_corr=[float(np.corrcoef(aggregate,x[:,k])[0,1]) for k in stage_cols]
            # Relative coefficient variance geometry, conditional on all GLM columns.
            gram_inv_diag=np.sum((vt[:rank].T/sing[:rank])**2,axis=1)
            uncentered_inflation=(x*x).sum(0)*gram_inv_diag
            target_inflation=uncentered_inflation[tc]
            implicated=[names[k] for k in np.flatnonzero(null_size>1e-6)]
            rec={'subject_id':run[0],'session_id':run[1],'run_id':run[2],
                'target_trials':len(rr),'design_columns':x.shape[1],'design_rank':rank,'rank_deficit':x.shape[1]-rank,
                'null_implicated_columns':' | '.join(implicated),'target_null_loading_max':float(null_size[tc].max()),
                'target_projection_error_max':float(target_estimability.max()),
                'response_feedback_design_correlation':float(stage_corr[1,2]),
                'valuation_sum_rating_design_correlation':val_stage_corr[0],
                'valuation_sum_response_design_correlation':val_stage_corr[1],
                'valuation_sum_feedback_design_correlation':val_stage_corr[2],
                'target_variance_inflation_median':float(np.median(target_inflation)),
                'target_variance_inflation_max':float(target_inflation.max()),
                'cross_food_fold_beta_error_abs_correlation_max':float(np.max(np.abs(tcorr[different]))),
                'effective_residual_df':float(spm['xX']['erdf'])}
            run_results.append(rec)
            digest=sha(spm_path);mask_path=spm_path.parent/'mask.nii';mask_digest=sha(mask_path)
            assert digest==old_inputs[str(spm_path)]['spm_sha256']
            assert mask_digest==old_inputs[str(spm_path)]['mask_sha256']
            external_hashes.append({'path':str(spm_path),'sha256':digest,'mask_sha256':mask_digest,'event_sha256':sha(evpath)})
            for k,r in enumerate(rr):
                trial_results.append({**{name:r[name] for name in ['subject_id','session_id','run_id','trial_id','image_id']},
                    'duration_actual_SPM':float(np.atleast_1d(u_map[f'target_trial_{int(r["trial_id"]):04d}']['dur'])[0]),
                    'target_null_loading':float(null_size[tc[k]]),'target_projection_error':float(target_estimability[k]),
                    'target_variance_inflation':float(target_inflation[k])})
            if j%25==0:print(f'SPM audited: {j+1}/{len(grouped)}',flush=True)
    payload=hashlib.sha256();max_error=0.;n_checked=0
    for r in mapping:
        vals=roi_values(r['beta_file_local'],ref,slices,local)
        payload.update(r['beta_file_local'].encode());payload.update(vals.tobytes())
        selected=vals[supports[r['subject_id']]];assert np.isfinite(selected).all()
        k=tuple(r[name] for name in ['subject_id','session_id','run_id','trial_id'])
        max_error=max(max_error,abs(float(selected.mean())-old_means[k]));n_checked+=1
        if n_checked%2000==0:print(f'Beta ROI re-extracted: {n_checked}/{len(mapping)}',flush=True)
    assert max_error==0.
    assert payload.hexdigest()==previous['provenance']['beta_sphere_payload_sha256']
    write_csv(out/'run_design_diagnostics.csv',run_results);write_csv(out/'trial_estimability.csv',trial_results)
    summary={'status':'complete','created_at':datetime.now().astimezone().isoformat(),
        'runs':len(run_results),'trials':len(mapping),'actual_duration_max_error':max(duration_errors),
        'all_stages_onsets_durations_match_events':True,'all_target_pmods_absent':True,
        'actual_HRF_config_counts':dict(bf_configs),'SPM_and_mask_hashes_match_previous':True,
        'beta_ROI_reextracted_trials':n_checked,'ROI_mean_max_absolute_difference':max_error,
        'beta_ROI_payload_sha256':payload.hexdigest(),'payload_matches_previous':True,
        'rank_deficient_runs':sum(r['rank_deficit']>0 for r in run_results),
        'rank_deficit':stats([r['rank_deficit'] for r in run_results]),
        'target_null_loading':stats([r['target_null_loading'] for r in trial_results]),
        'target_projection_error':stats([r['target_projection_error'] for r in trial_results]),
        'target_variance_inflation':stats([r['target_variance_inflation'] for r in trial_results]),
        'response_feedback_correlation':stats([r['response_feedback_design_correlation'] for r in run_results]),
        'cross_food_fold_max_beta_error_abs_correlation':stats([r['cross_food_fold_beta_error_abs_correlation_max'] for r in run_results]),
        'null_implicated_names':dict(Counter(n for r in run_results for n in r['null_implicated_columns'].split(' | ') if n)),
        'external_input_hashes':external_hashes,'mapping_sha256':sha(mapping_path),'script_sha256':sha(__file__),
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'nibabel':nib.__version__},
        'seed':None,'no_data_exclusions_or_GLM_refits':True,
        'diagnostic_definitions':{'target_projection_error':'norm of column e_j of pKX @ filtered_whitened_X minus identity; numerical estimability only',
            'target_variance_inflation':'uncentered column squared norm times diagonal of effective-design pseudoinverse Gram; conditional geometry, not ROI reliability',
            'beta_error_correlation':'normalized SPM Bcov; expected coefficient-error correlation under the fitted noise model, not measured neural similarity'},
        'output_sha256':{p.name:sha(p) for p in out.glob('*.csv')}}
    (out/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');shutil.copy2(__file__,out/'script_snapshot.py')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['external_input_hashes','null_implicated_names']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
