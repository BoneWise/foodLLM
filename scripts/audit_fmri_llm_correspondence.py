"""Read-only provenance audit from raw task events to beta, ROI and model image.

Records current external-file availability separately from the previously
completed SPM audit. Never refits GLMs or alters historical inputs.
"""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np


def rows(path,delimiter=','):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f,delimiter=delimiter))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def key(r):
    return tuple(r[k] for k in ['subject_id','session_id','run_id','trial_id'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files={'ratings':'data/prepared/trial_ratings_with_nutrition.csv',
           'mapping':'data/external/behavior_to_singletrial_beta_29subjects.csv',
           'beta_index':'archive/handoff_20260915/data/original_beta_index_31subjects.csv',
           'roi':'outputs/20260916_vmPFC准备_v01/roi_trial_responses.csv',
           'previous_spm_audit':'outputs/20260916_vmPFC准备_v01/audit.json',
           'records':'outputs/20260916_Qwen_AB_features_v03/records.json',
           'images':'data/prepared/image_files_568.csv',
           'lsa_source':'archive/source_code/run_minimal_lsa.m'}
    ratings=rows(files['ratings']);mapping=rows(files['mapping']);roi=rows(files['roi'])
    mm={key(r):r for r in mapping};rm={key(r):r for r in roi}
    assert len(mm)==len(mapping)==len(rm)==len(roi)==len(ratings)==15869
    assert set(mm)==set(rm)=={key(r) for r in ratings}
    original=rows(files['beta_index'])
    om={(r['sub'],r['session'],r['run'].removeprefix('run-'),r['trial_in_run']):r for r in original}
    assert len(om)==len(original)
    previous=json.loads(Path(files['previous_spm_audit']).read_text())
    assert sha(files['mapping'])==previous['provenance']['mapping_sha256']
    assert previous['design_errors']==[] and previous['runs']==346
    event_cache={};event_hashes={}
    table_root=Path('outputs/20260916_分析输入索引_v01')
    idx=json.loads((table_root/'index.json').read_text())
    subjects={s:i for i,s in enumerate(idx['subject_ids'])};images={s:i for i,s in enumerate(idx['image_ids'])}
    arrays={name:np.load(table_root/(name+'.npy')) for name in ['human_ratings','roi_beta','session_ids','run_ids','primary_mask']}
    local_images={r['image_id']:r['image_path'] for r in rows(files['images'])}
    image_hashes={i:sha(p) for i,p in local_images.items()}
    records=json.loads(Path(files['records']).read_text())
    assert len(records)==len({(r['prompt_id'],r['image_id'],r['condition']) for r in records})==7202
    for r in records:
        assert r['image_sha256']==image_hashes[r['image_id']]
        assert int(Path(r['image_path']).stem)==int(r['image_id'])
        assert r['readout_index']==r['token_count']-1 and r['enable_thinking'] is False
    matrix_root=Path('outputs/20260916_Qwen_AB分析矩阵_v01');matrix_rows=0
    primary_ids={i for i,j in images.items() if arrays['primary_mask'][j]}
    for prompt in ['A','B']:
        for c in ['V','T','P1','P2','P3','P4','P5']:
            indexed=rows(matrix_root/f'{prompt}_{c}'/'image_index.csv')
            assert {r['image_id'] for r in indexed}==(set(images) if c in ['V','T'] else primary_ids)
            for r in indexed:
                assert r['prompt_id']==prompt and r['condition']==c
                assert r['source_cache']==f'{prompt}_{r["image_id"]}_{c}.npz'
            matrix_rows+=len(indexed)
    chain=[];phase_counts=Counter();duration=[];all_stages=Counter()
    for r in ratings:
        k=key(r);m=mm[k];b=rm[k];o=om[k]
        path=r['events_file']
        if path not in event_cache:
            ev=rows(path,'\t');event_hashes[path]=sha(path)
            lookup={}
            for e in ev:
                lookup.setdefault((e['trial_id'],e['trial_type']),[]).append(e)
            event_cache[path]=lookup
        matches=event_cache[path][(r['trial_id'],'valuation')]
        assert len(matches)==1;e=matches[0]
        assert e['image_id']==r['image_id']==m['image_id']==b['image_id']==o['stimulus_id']
        assert o['trial_type']=='valuation'
        expected_regressor=f'target_trial_{int(r["trial_id"]):04d}'
        assert o['regressor_name']==expected_regressor
        assert Path(o['beta_file']).name==Path(m['beta_file_local']).name
        assert o['beta_file']==m['beta_file'] and o['spm_mat']==m['spm_mat']
        assert o['estimated_successfully']=='1'
        assert max(abs(float(e['onset'])-float(x)) for x in [r['onset'],m['onset_behavior'],m['onset_beta'],o['onset']])<1e-6
        assert max(abs(float(e['duration'])-float(x)) for x in [r['valuation_duration'],o['duration']])<1e-6
        assert float(e['rating_value'])==float(r['rating'])==float(m['rating'])==float(o['rating_value'])
        si,ii=subjects[r['subject_id']],images[r['image_id']]
        assert arrays['human_ratings'][si,ii]==float(r['rating'])
        assert arrays['roi_beta'][si,ii]==float(b['roi_beta_mean'])
        assert arrays['session_ids'][si,ii]==r['session_id']
        assert arrays['run_ids'][si,ii]==r['session_id']+'/'+r['run_id']
        phase_counts[o['trial_type']]+=1;duration.append(float(e['duration']))
        for stage in ['rating','response','feedback']:
            es=event_cache[path].get((r['trial_id'],stage),[])
            assert len(es)==1
            assert es[0]['image_id']==r['image_id']
            all_stages[stage]+=1
        rating_event=event_cache[path][(r['trial_id'],'rating')][0]
        assert abs(float(rating_event['onset'])-(float(e['onset'])+float(e['duration'])))<1e-6
        chain.append({**dict(zip(['subject_id','session_id','run_id','trial_id'],k)),
            'image_id':r['image_id'],'phase':o['trial_type'],'valuation_onset':e['onset'],
            'valuation_duration':e['duration'],'rating_onset':rating_event['onset'],
            'SPM_expected_column':f'Sn(1) {expected_regressor}*bf(1)',
            'beta_file':m['beta_file_local'],'roi_beta_mean':b['roi_beta_mean'],
            'analysis_subject_row':si,'analysis_image_column':ii,'primary_493':bool(arrays['primary_mask'][ii]),
            'model_image_path_local':local_images[r['image_id']],
            'model_input_image_sha256':image_hashes[r['image_id']]})
    assert len({(r['subject_id'],r['image_id']) for r in ratings})==15869
    spm_paths=sorted({r['spm_mat_local'] for r in mapping})
    available=sum(Path(p).exists() for p in spm_paths)
    with (out/'trial_correspondence.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(chain[0]));w.writeheader();w.writerows(chain)
    report={'status':'local_chain_passed_external_reread_unavailable' if available==0 else 'local_chain_passed',
        'created_at':datetime.now().astimezone().isoformat(),'subjects':len(subjects),
        'images':len(images),'trials_checked':len(chain),'runs_checked_from_local_events':len(event_cache),
        'primary_images':len(primary_ids),'primary_trials':sum(r['primary_493'] for r in chain),
        'phase_counts':dict(phase_counts),'later_stage_event_counts':dict(all_stages),
        'valuation_duration_seconds':{'min':min(duration),'median':float(np.median(duration)),
            'max':max(duration),'quantiles_1_99':np.quantile(duration,[.01,.99]).tolist()},
        'model_input_records_image_hash_verified':len(records),'unique_model_images_hash_verified':len(image_hashes),
        'matrix_rows_ID_verified':matrix_rows,'current_external_SPM_files_available':available,
        'historical_SPM_audit':{'created_at':previous['created_at'],'runs':previous['runs'],
            'trials':previous['trials'],'design_errors':previous['design_errors'],
            'mapping_hash_still_matches':True,'checks':'actual SPM target column, onset, absence of target pmod',
            'duration_in_actual_SPM_was_not_checked_by_previous_audit':True},
        'current_duration_evidence':'all local raw events agree with original beta index and prepared behavioral table; LSA code copies valuation duration',
        'example':chain[0],
        'limitations':['Current external SPM and beta volumes not re-read when unavailable.',
            'Correct identity linkage does not prove functional reliability or model-brain predictive alignment.',
            'Single ROI mean beta per image/subject; no time-resolved or full voxel-pattern alignment.',
            'All model layers and V/T/P conditions target the same image-only human valuation beta.'],
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'seed':None,'dependencies':{'python':platform.python_version(),'numpy':np.__version__},
        'input_sha256':{k:sha(p) for k,p in files.items()},'event_sha256':event_hashes,
        'array_sha256':{p.name:sha(p) for p in table_root.glob('*.npy')},
        'script_sha256':sha(__file__),'output_chain_sha256':sha(out/'trial_correspondence.csv')}
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(__file__,out/'script_snapshot.py')
    print(json.dumps({k:v for k,v in report.items() if k not in ['input_sha256','event_sha256','array_sha256']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
