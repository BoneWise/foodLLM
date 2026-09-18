"""Index existing prepared ratings/ROI means for modeling; no new cleaning or fitting."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True)
    ap.add_argument('--preparation-config',help='Optional versioned preparation specification')
    args=ap.parse_args();out=ROOT/args.output
    assert out.resolve().is_relative_to(ROOT/'outputs');out.mkdir(parents=True,exist_ok=False)
    project=json.loads((ROOT/'configs/project.json').read_text())
    spec_path=ROOT/(args.preparation_config or project['current_preparation_config'])
    spec=json.loads(spec_path.read_text())
    eligibility=spec['neural'].get('analysis_eligibility_column','geometry_and_finite_eligible')
    files={'ratings':ROOT/project['paths']['trial_ratings'],
           'images':ROOT/project['paths']['image_summary'],
           'folds':ROOT/spec['splits']['fold_file'],
           'brain':ROOT/spec['neural']['response_table']}
    ratings=rows(files['ratings']);brain=rows(files['brain'])
    meta={r['image_id']:r for r in rows(files['images'])}
    fold_rows=sorted(rows(files['folds']),key=lambda r:int(r['image_id']))
    images=[r['image_id'] for r in fold_rows];subjects=sorted({r['subject_id'] for r in ratings})
    assert len(images)==len(set(images))==568 and len(subjects)==29
    assert 'sub-004' not in subjects and 'sub-010' not in subjects and 'sub-016' in subjects
    ii={s:i for i,s in enumerate(images)};si={s:i for i,s in enumerate(subjects)}
    def trial_key(r):
        return tuple(r[k] for k in ['subject_id','session_id','run_id','trial_id'])
    assert len(ratings)==len(brain)==15869
    assert len({trial_key(r) for r in ratings})==len(ratings)
    brain_map={trial_key(r):r for r in brain}
    assert len(brain_map)==len(brain) and set(brain_map)=={trial_key(r) for r in ratings}
    human=np.full((29,568),np.nan);beta=human.copy()
    sessions=np.full((29,568),'',dtype='<U16');runs=np.full((29,568),'',dtype='<U32')
    for r in ratings:
        b=brain_map[trial_key(r)];assert b['image_id']==r['image_id']
        s,i=si[r['subject_id']],ii[r['image_id']]
        assert np.isnan(human[s,i]), 'Unexpected repeated subject-image observation'
        human[s,i]=float(r['rating']);assert human[s,i] in [1.,2.,3.,4.]
        assert b[eligibility].lower() in ['true','false']
        if b[eligibility].lower()=='true':
            assert b['geometry_and_finite_eligible'].lower()=='true'
            beta[s,i]=float(b['roi_beta_mean']);assert np.isfinite(beta[s,i])
        sessions[s,i]=r['session_id'];runs[s,i]=r['session_id']+'/'+r['run_id']
    columns=['red','green','blue','object_size','log1p(grams_total)','correct_kcal_100g']
    content=[]
    for i in images:
        r=meta[i];grams=float(r['grams_total']);assert grams>=0
        content.append([*[float(r[k]) for k in columns[:4]],np.log1p(grams),float(r['kcal_100g'])])
    content=np.asarray(content);assert np.isfinite(content).all()
    primary=np.array([r['P_eligible']=='True' for r in fold_rows])
    folds=np.array([int(r['outer_fold']) for r in fold_rows])
    assert primary.sum()==493 and np.isfinite(human[:,primary]).sum()==13767
    assert np.isfinite(beta).sum()==spec['neural']['finite_eligible_trials']
    if 'primary_eligible_trials' in spec['neural']:
        assert np.isfinite(beta[:,primary]).sum()==spec['neural']['primary_eligible_trials']
    # Reconcile existing prepared means; this is an identity check, not re-cleaning.
    assert np.allclose(np.nanmean(human,axis=0),[float(meta[i]['mean_rating']) for i in images],atol=1e-12)
    arrays={'human_ratings':human,'roi_beta':beta,'session_ids':sessions,'run_ids':runs,
            'content_covariates':content,'primary_mask':primary,'outer_folds':folds}
    for name,value in arrays.items():
        np.save(out/(name+'.npy'),value,allow_pickle=False)
    index={'image_ids':images,'subject_ids':subjects,'food_groups':[r['food_group'] for r in fold_rows],
        'content_columns':columns,'axes_subject_image':['subject','image'],
        'missing_values':'NaN for unobserved ratings or unobserved/excluded beta; session/run retained for observed behavior',
        'run_identity':'session_id/run_id; original run leading zeros retained'}
    (out/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    report={'status':'complete','subjects':29,'images':568,'original_ratings':15869,
        'primary_images':493,'primary_observations':13767,'new_exclusions':0,
        'brain_observations':int(np.isfinite(beta).sum()),
        'primary_brain_observations':int(np.isfinite(beta[:,primary]).sum()),
        'brain_exclusions_from_original':int(15869-np.isfinite(beta).sum()),
        'behavior_exclusions_added':0,'eligibility_column':eligibility,
        'preparation_config':str(spec_path.relative_to(ROOT)),
        'preparation_config_sha256':digest(spec_path),
        'no_new_cleaning_or_statistical_fit':True,'subject_image_alignment_and_mean_identity_verified':True,
        'all_brain_eligibility_from_existing_rules':True,'numpy_version':np.__version__,
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'script_sha256':digest(__file__),'input_sha256':{k:digest(v) for k,v in files.items()},
        'outputs':{p.name:digest(p) for p in sorted(out.iterdir())}}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','subjects','images','primary_images','primary_observations','new_exclusions']}))


if __name__=='__main__':
    main()
