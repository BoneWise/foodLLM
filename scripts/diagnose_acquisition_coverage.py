"""Flag valuation epochs not fully inside the recorded run; do not change data."""
import argparse,csv,json,shutil,subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from prepare_roi import sha,write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with Path('data/external/behavior_to_singletrial_beta_29subjects.csv').open() as f:rr=list(csv.DictReader(f))
    with Path('outputs/20260916_vmPFC准备_v01/roi_trial_responses.csv').open() as f:
        means={tuple(r[k] for k in ['subject_id','session_id','run_id','trial_id']):float(r['roi_beta_mean']) for r in csv.DictReader(f)}
    grouped=defaultdict(list)
    for r in rr:grouped[r['spm_mat_local']].append(r)
    rows=[]
    for path,records in sorted(grouped.items()):
        s=loadmat(path,simplify_cells=True)['SPM'];end=int(s['nscan'])*float(s['xY']['RT'])
        names=list(s['xX']['name']);bcov=np.asarray(s['xX']['Bcov']);x=np.asarray(s['xX']['X'])
        cols=[names.index(f'Sn(1) target_trial_{int(r["trial_id"]):04d}*bf(1)') for r in records]
        medvar=np.median(np.diag(bcov)[cols])
        um={str(np.atleast_1d(u['name'])[0]):u for u in s['Sess']['U']}
        for r,col in zip(records,cols):
            u=um[f'target_trial_{int(r["trial_id"]):04d}'];onset=float(u['ons']);duration=float(u['dur']);offset=onset+duration
            key=tuple(r[k] for k in ['subject_id','session_id','run_id','trial_id'])
            rows.append({**dict(zip(['subject_id','session_id','run_id','trial_id'],key)),
                'image_id':r['image_id'],'valuation_onset':onset,'valuation_end':offset,
                'scan_nominal_end':end,'valuation_fully_inside_acquisition':onset>=0 and offset<=end,
                'seconds_after_valuation_end':end-offset,
                'observed_valuation_duration_fraction':max(0.,min(end,offset)-max(0.,onset))/duration,
                'design_column_norm':float(np.linalg.norm(x[:,col])),
                'Bcov_diagonal':float(bcov[col,col]),'Bcov_diagonal_vs_run_target_median':float(bcov[col,col]/medvar),
                'roi_beta_mean':means[key]})
    flags=[r for r in rows if not r['valuation_fully_inside_acquisition']]
    write_csv(out/'trial_acquisition_coverage.csv',rows);write_csv(out/'incomplete_valuation_trials.csv',flags)
    summary={'status':'complete','trials':len(rows),'incomplete_valuation_trials':flags,
        'criterion':'valuation onset >= 0 and onset + duration <= nscan * TR; post-estimate timing diagnostic, not outcome-based exclusion',
        'tail_counts_descriptive_not_exclusion_thresholds':{str(t):sum(r['seconds_after_valuation_end']<t for r in rows) for t in [0,4,8,16]},
        'limitations':'A fully observed valuation epoch does not guarantee enough post-event HRF coverage or high beta precision.',
        'beta_variance_note':'SPM Bcov is coefficient covariance up to residual variance; not a direct ROI standard error.',
        'seed':None,'script_sha256':sha(__file__),
        'input_SPM_audit_sha256':sha('outputs/20260916_实际SPM与beta复核_v01/audit.json'),
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'output_sha256':{p.name:sha(p) for p in out.glob('*.csv')}}
    (out/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');shutil.copy2(__file__,out/'script_snapshot.py')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
