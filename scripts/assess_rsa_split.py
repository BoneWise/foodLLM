"""Audit existing repetition and split support; no RSA or new model fitting."""
import csv
import json
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from run_layer_analysis import sha


def main():
    out=Path('outputs/20260916_RSA划分与MSE后续评估_v01')
    out.mkdir(exist_ok=False)
    config=Path('configs/project.json');project=json.loads(config.read_text())
    root=Path(project['paths']['analysis_tables']);ratings=Path(project['paths']['trial_ratings'])
    with ratings.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    counts=Counter((r['subject_id'],r['image_id']) for r in rows)
    assert len(rows)==15869 and max(counts.values())==1
    index=json.loads((root/'index.json').read_text())
    beta=np.load(root/'roi_beta.npy');sessions=np.load(root/'session_ids.npy')
    primary=np.load(root/'primary_mask.npy');folds=np.load(root/'outer_folds.npy')
    report={'status':'complete','scope':'split_support_audit_no_RSA_or_inference',
        'subject_image_repeated_pairs':sum(v>1 for v in counts.values()),
        'maximum_presentations_per_subject_image':max(counts.values()),
        'sessions':sorted(set(sessions[sessions!=''])),'subjects':len(index['subject_ids']),
        'primary_fold_image_counts':[int(np.sum(primary&(folds==k))) for k in sorted(set(folds))],
        'samples':{}}
    for label,mask in [('primary',primary),('full_VT',np.ones(568,dtype=bool))]:
        valid=np.isfinite(beta)&mask;sizes=[];subject_sessions=[]
        for s in range(len(index['subject_ids'])):
            ss=sorted(set(sessions[s,valid[s]]));subject_sessions.append(len(ss))
            for session in ss:
                for k in sorted(set(folds)):
                    train=valid[s]&(folds!=k)&(sessions[s]!=session)
                    test=valid[s]&(folds==k)&(sessions[s]==session)
                    assert train.sum()>0 and test.sum()>0
                    sizes.append((int(train.sum()),int(test.sum())))
        report['samples'][label]={'images':int(mask.sum()),'brain_observations':int(valid.sum()),
            'per_subject_images_min':int(valid.sum(1).min()),'per_subject_images_max':int(valid.sum(1).max()),
            'images_valid_in_all_29':int(valid.all(0).sum()),
            'subjects_per_image_min':int(valid[:,mask].sum(0).min()),
            'subjects_per_image_max':int(valid[:,mask].sum(0).max()),
            'sessions_per_subject_min':min(subject_sessions),'sessions_per_subject_max':max(subject_sessions),
            'food_session_cells':len(sizes),'train_size_range':[min(x[0] for x in sizes),max(x[0] for x in sizes)],
            'test_size_range':[min(x[1] for x in sizes),max(x[1] for x in sizes)]}
    (out/'split_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(__file__,out/'script_snapshot.py')
    inputs=[config,ratings,root/'index.json']+[root/(k+'.npy') for k in ['roi_beta','session_ids','primary_mask','outer_folds']]
    audit={'status':'complete','input_sha256':{str(p):sha(p) for p in inputs},
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dependencies':{'python':platform.python_version(),'numpy':np.__version__},
        'seed':None,'randomness':'none','script_sha256':sha(__file__),
        'validation':'unique subject-image records; all crossed brain train/test cells nonempty',
        'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
