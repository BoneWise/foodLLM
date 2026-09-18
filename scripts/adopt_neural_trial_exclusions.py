"""Adopt the two user-approved brain-only exclusions; preserve all prior artifacts.

Reuse unchanged behavior/q and validated timing-sensitivity predictions. Verify
the affected participant's fits again against the new input before publishing.
Project entry points are switched separately, after validation.
"""
import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from feature_inputs import load_layer_inputs
from run_layer_analysis import brain_from_q, sha

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['subject_id', 'session_id', 'run_id', 'trial_id', 'image_id']
DATA = Path('outputs/20260916_脑试次剔除_v01')
TABLES = Path('outputs/20260916_分析输入索引_v02')
RESULTS = Path('outputs/20260916_逐层行为与脑预测_v02')
SUMMARY = Path('outputs/20260916_逐层结果汇总_v02')
OLD = Path('outputs/20260916_逐层行为与脑预测_v01')
SENS = Path('outputs/20260916_截断试次脑敏感性_v01')


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def key(row):
    return tuple(row[k] for k in KEYS)


def main():
    assert Path.cwd() == ROOT
    flags_path = Path('outputs/20260916_估值事件采集覆盖诊断_v01/incomplete_valuation_trials.csv')
    source = Path('outputs/20260916_vmPFC准备_v01/roi_trial_responses.csv')
    flags = read_rows(flags_path)
    expected = {('sub-002','ses-01','04','48','443'), ('sub-002','ses-02','04','48','43')}
    assert {key(r) for r in flags} == expected and len(flags) == 2
    DATA.mkdir(exist_ok=False)
    base = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    exclusion_cfg = Path('configs/neural_trial_exclusions_v01.json')
    policy = {
        'version':'v01', 'decision_date':'2026-09-16',
        'authorization':'User explicitly requested: 把这两个试次剔除掉',
        'scope':'brain_observations_only', 'reason':'valuation_event_ends_after_acquisition_end',
        'exact_match_columns':KEYS, 'excluded_trials':[dict(zip(KEYS,k)) for k in sorted(expected)],
        'behavior_ratings_retained':True, 'exclude_entire_subject_or_image':False,
        'decision_after_effects_visible':True, 'prospective_preregistration':False,
        'general_HRF_tail_or_precision_threshold_adopted':False,
        'evidence':str(flags_path), 'evidence_sha256':sha(flags_path),
        'expected_brain_counts':{'full':15867,'primary':13765,'subjects':29},
        'seed':None, 'selection':'exact_two_audited_trials_no_random_selection'
    }
    save(exclusion_cfg,policy)
    original = read_rows(source)
    result = []
    for r in original:
        excluded = key(r) in expected
        result.append({**r,'neural_analysis_eligible':str(r['geometry_and_finite_eligible'].lower()=='true' and not excluded),
                       'neural_exclusion_reason':policy['reason'] if excluded else ''})
    excluded = [r for r in result if key(r) in expected]
    included = [r for r in result if r['neural_analysis_eligible']=='True']
    assert len(result)==15869 and len(excluded)==2 and len(included)==15867
    write_rows(DATA/'roi_trial_responses.csv',result)
    write_rows(DATA/'roi_trial_responses_included.csv',included)
    write_rows(DATA/'excluded_trials.csv',excluded)
    write_rows(DATA/'timing_evidence.csv',flags)
    prep_path=Path('configs/nutrient_layer_alignment_v05_prepared_v04.json')
    prep=json.loads(Path('configs/nutrient_layer_alignment_v05_prepared_v03.json').read_text())
    prep['status']='v05_descriptive_estimates_two_neural_exclusions_adopted_inference_pending'
    prep['neural'].update(response_table=str(DATA/'roi_trial_responses.csv'),
        analysis_eligibility_column='neural_analysis_eligible', exclusion_policy=str(exclusion_cfg),
        finite_eligible_trials=15867, primary_eligible_trials=13765,
        quality_scope='geometry_finite_and_two_user_approved_timing_exclusions_not_reliability')
    prep['implementation_status'].update(analysis_tables=str(TABLES),
        descriptive_execution_config='configs/layer_estimation_v02.json',
        descriptive_estimates=str(RESULTS),summary=str(SUMMARY),
        human_brain_positive_control='outputs/20260916_人类wanting脑正对照_时窗敏感性_v01')
    prep['analysis_status']['neural_exclusions']='two_exact_trials_user_approved_after_effects_visible'
    prep['preparation_record']='docs/research/20260916_两条截断试次正式剔除记录.md'
    save(prep_path,prep)
    cfg=json.loads(Path('configs/layer_estimation_v01.json').read_text())
    cfg.update(version='v02',status='two_user_approved_brain_exclusions_after_effects_visible',
        parent_config=str(prep_path),output=str(RESULTS),analysis_root=str(TABLES),
        neural_exclusion_policy=str(exclusion_cfg),
        adoption_method='reuse_unchanged_behavior_and_q_adopt_verified_sensitivity_brain_refits')
    save('configs/layer_estimation_v02.json',cfg)
    subprocess.run([sys.executable,'scripts/prepare_analysis_tables.py','--output',str(TABLES),
                    '--preparation-config',str(prep_path)],check=True)
    previous=Path('outputs/20260916_分析输入索引_v01')
    for p in TABLES.glob('*.npy'):
        if p.name!='roi_beta.npy': assert sha(p)==sha(previous/p.name)
    before=np.load(previous/'roi_beta.npy');after=np.load(TABLES/'roi_beta.npy')
    changed=~((before==after)|(np.isnan(before)&np.isnan(after)))
    index=json.loads((TABLES/'index.json').read_text())
    actual={(index['subject_ids'][s],index['image_ids'][i]) for s,i in np.argwhere(changed)}
    assert actual=={(k[0],k[4]) for k in expected}
    assert np.isnan(after[changed]).all()
    save(DATA/'validation.json',{'status':'passed','brain_entries_changed':2,
        'all_other_analysis_arrays_byte_identical':True,'behavior_rows_retained':15869,
        'brain_rows_full':15867,'brain_rows_primary':13765,'subject_and_image_ids_unchanged':True})
    shutil.copy2(__file__,DATA/'script_snapshot.py')
    save(DATA/'audit.json',{'status':'complete','git_base':base,'seed':None,
        'dependencies':{'python':platform.python_version(),'numpy':np.__version__},
        'input_sha256':{str(p):sha(p) for p in [source,flags_path,exclusion_cfg,prep_path]},
        'script_sha256':sha(__file__),
        'output_sha256':{p.name:sha(p) for p in DATA.iterdir() if p.is_file()}})

    old_audit=json.loads((OLD/'audit.json').read_text())
    sens_audit=json.loads((SENS/'audit.json').read_text())
    assert old_audit['status']==sens_audit['status']=='complete'
    assert json.loads((SENS/'validation.json').read_text())['status']=='passed'
    RESULTS.mkdir(exist_ok=False)
    shutil.copy2('configs/layer_estimation_v02.json',RESULTS/'execution_config.json')
    shutil.copy2(__file__,RESULTS/'adoption_script.py')
    rows=[]; max_error=0.; verified=0
    with threadpool_limits(limits=2):
        for sample in cfg['samples']:
            for prompt in cfg['prompts']:
                data=load_layer_inputs(cfg['matrix_root'],str(TABLES),prompt,1,sample)
                observed=np.isfinite(data['roi_beta']);s=data['subject_ids'].index('sub-002')
                for layer in cfg['layers']:
                    name=f'{sample}_{prompt}_layer{layer:02d}'
                    for suffix in ['.json','.npz']:
                        assert sha(OLD/(name+suffix))==old_audit['output_sha256'][name+suffix]
                    assert sha(SENS/(name+'.npz'))==sens_audit['output_sha256'][name+'.npz']
                    with np.load(OLD/(name+'.npz')) as src:
                        arrays={k:src[k] for k in src.files}
                    meta=json.loads((OLD/(name+'.json')).read_text())
                    assert meta['rank_deficient_brain_fits']==0
                    with np.load(SENS/(name+'.npz')) as corrected:
                        for k in ['brain_predictions','brain_squared_errors']:
                            unchanged=np.arange(29)!=s
                            assert np.array_equal(arrays[k][:,unchanged],corrected[k][:,unchanged],equal_nan=True)
                            arrays[k]=corrected[k]
                    # Verify adopted fits against the newly indexed data and confirm rank metadata.
                    ranks_bad=0
                    for fold in sorted(set(data['outer_folds'])):
                        train=np.flatnonzero(data['outer_folds']!=fold);test=np.flatnonzero(data['outer_folds']==fold)
                        q={c:arrays[f'q_fold{fold}_{c}'][:,s] for c in meta['conditions']}
                        for mode in range(2):
                            sessions=[None] if mode==0 else sorted(set(data['session_ids'][s,observed[s]]))
                            for session in sessions:
                                b,p,ranks=brain_from_q(q,data['roi_beta'][s],data['content_covariates'],
                                    data['raw_V_probs'],train,test,data['run_ids'][s],data['session_ids'][s],session)
                                for cidx,c in enumerate(meta['brain_conditions']):
                                    err=float(np.max(np.abs(p[c]-arrays['brain_predictions'][mode,s,cidx,b])))
                                    max_error=max(max_error,err);assert err<1e-10
                                    ranks_bad+=int(ranks[c][0]<ranks[c][1])
                    meta['rank_deficient_brain_fits']=ranks_bad
                    assert np.array_equal(np.isfinite(arrays['brain_predictions']),
                        np.broadcast_to(observed[None,:,None,:],arrays['brain_predictions'].shape))
                    assert np.array_equal(arrays['brain_squared_errors'],
                        (arrays['brain_predictions']-data['roi_beta'][None,:,None,:])**2,equal_nan=True)
                    arrays['brain_subject_mse']=np.nanmean(arrays['brain_squared_errors'],axis=-1)
                    arrays['test_counts']=np.broadcast_to(observed,(2,*observed.shape)).astype(np.int8).copy()
                    meta['observed_trials']=int(observed.sum())
                    meta['adoption']={'brain_source':str(SENS/(name+'.npz')),
                        'unchanged_behavior_and_q_source':str(OLD/(name+'.npz')),
                        'exclusion_policy':str(exclusion_cfg),'not_a_fresh_upstream_refit':True}
                    meta['original_elapsed_seconds']=meta.pop('elapsed_seconds',None)
                    for mi,mode in enumerate(meta['brain_modes']):
                        m=dict(zip(meta['brain_conditions'],arrays['brain_subject_mse'][mi].mean(0).tolist()))
                        m.update(G=m['V']-m['T'],T_vs_baseline=m['baseline']-m['T'])
                        if len(meta['conditions'])>2:
                            m['P_mean']=float(np.mean([m[c] for c in meta['conditions'] if c.startswith('P')]))
                            m['S']=m['P_mean']-m['T']
                        meta['metrics'][mode]=m
                    np.savez_compressed(RESULTS/(name+'.npz'),**arrays)
                    save(RESULTS/(name+'.json'),meta)
                    for endpoint,m in meta['metrics'].items():
                        rows.append({'sample':sample,'prompt':prompt,'layer':layer,'endpoint':endpoint,**m})
                    verified+=1
                print(f'Adopted and verified {sample} {prompt}',flush=True)
    fields=['sample','prompt','layer','endpoint','baseline','V','T','P1','P2','P3','P4','P5','P_mean','G','S','T_vs_baseline']
    with (RESULTS/'layer_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    save(RESULTS/'adoption_validation.json',{'status':'passed','tasks_checked':verified,
        'affected_subject_refit_max_absolute_error':max_error,'unaffected_28_subjects_exactly_preserved':True,
        'behavior_and_upstream_q_reused_unchanged':True,'brain_eligibility_matches_new_input':True,
        'full_and_primary_observations':[15867,13765],'inference_complete':False})
    save(RESULTS/'audit.json',{'status':'complete','completed':old_audit['completed'],
        'scope':'descriptive_estimates_after_user_approved_brain_only_exclusions',
        'adoption_method':cfg['adoption_method'],'git_base':base,'seed':None,
        'bootstrap_performed':False,'config_sha256':sha('configs/layer_estimation_v02.json'),
        'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'input_sha256':{str(p):sha(p) for p in [OLD/'audit.json',SENS/'audit.json',SENS/'validation.json',
                                             TABLES/'audit.json',DATA/'audit.json',exclusion_cfg]},
        'script_sha256':sha(__file__),
        'output_sha256':{p.name:sha(p) for p in RESULTS.iterdir() if p.is_file()}})
    print(json.dumps({'tasks':verified,'refit_max_error':max_error}))


if __name__=='__main__':
    main()
