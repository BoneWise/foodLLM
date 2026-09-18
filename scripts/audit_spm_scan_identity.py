"""Verify actual SPM scan identities and volume/metadata alignment, read-only."""
import argparse,csv,json,platform,shutil,subprocess
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from prepare_roi import sha,write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with Path('data/external/behavior_to_singletrial_beta_29subjects.csv').open() as f:rows=list(csv.DictReader(f))
    grouped={}
    for r in rows:grouped.setdefault((r['subject_id'],r['session_id'],r['run_id']),r['spm_mat_local'])
    results=[]
    for i,(key,path) in enumerate(sorted(grouped.items())):
        s=loadmat(path,simplify_cells=True)['SPM'];sub,ses,run=key
        scans=[str(v).strip().rsplit(',',1) for v in np.atleast_1d(s['xY']['P'])]
        files={v[0] for v in scans};assert len(files)==1
        bold=Path(scans[0][0]);stem=f'{sub}_{ses}_task-food_run-{run}'
        assert bold.name.startswith(stem+'_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold')
        assert [int(v[1]) for v in scans]==list(range(1,len(scans)+1))
        image=nib.load(bold);n=len(scans)
        assert image.shape[3]==n==int(s['nscan'])==s['xX']['X'].shape[0]
        assert abs(float(image.header.get_zooms()[3])-float(s['xY']['RT']))<1e-6
        assert np.array_equal(np.asarray(s['Vbeta'][0]['dim']),image.shape[:3])
        # SPM mat maps one-based voxel coordinates; nibabel uses zero-based.
        zero_to_one=np.eye(4);zero_to_one[:3,3]=1
        assert np.allclose(np.asarray(s['Vbeta'][0]['mat'])@zero_to_one,image.affine,atol=1e-5)
        func=Path('/path/to/external_data/fmriPrep_31out')/sub/ses/'func'
        conf=func/f'{stem}_desc-confounds_timeseries.tsv'
        with conf.open() as f:n_conf=sum(1 for _ in csv.DictReader(f,delimiter='\t'))
        assert n_conf==n
        external_events=Path('/path/to/external_data/fmriPrep_31out/events')/sub/ses/'func'/f'{stem}_events.tsv'
        local_events=Path('data/raw/events')/sub/ses/'func'/f'{stem}_events.tsv'
        assert sha(external_events)==sha(local_events)
        with local_events.open() as f:ev=list(csv.DictReader(f,delimiter='\t'))
        last_event=max(float(e['onset'])+float(e['duration']) for e in ev)
        scan_end=n*float(s['xY']['RT'])
        valuation=[e for e in ev if e['trial_type']=='valuation']
        last_valuation=max(float(e['onset'])+float(e['duration']) for e in valuation)
        valuation_outside=sum(float(e['onset'])+float(e['duration'])>scan_end for e in valuation)
        results.append({'subject_id':sub,'session_id':ses,'run_id':run,'volumes':n,
            'nifti_TR':float(image.header.get_zooms()[3]),'SPM_TR':float(s['xY']['RT']),
            'first_volume':1,'last_volume':n,'confound_rows':n_conf,'last_event_end_seconds':last_event,
            'scan_duration_seconds':scan_end,'last_event_end_beyond_scan':last_event>scan_end,
            'last_valuation_end_seconds':last_valuation,'valuation_events_ending_beyond_scan':valuation_outside,
            'scan_tail_after_last_valuation_seconds':scan_end-last_valuation,'input_bold':str(bold),
            'input_bold_header_sha256':__import__('hashlib').sha256(image.header.binaryblock).hexdigest(),
            'SPM_sha256':sha(path),'confound_sha256':sha(conf),'events_sha256':sha(local_events),
            'affine_consistent_with_beta':True})
        if i%50==0:print(f'Scan identity checked: {i+1}/{len(grouped)}',flush=True)
    write_csv(out/'run_scan_identity.csv',results)
    a={'status':'complete','runs':len(results),'all_run_subject_session_identifiers_match':True,
       'all_volume_sequences_start_at_one_and_complete':True,'all_nifti_SPM_confound_counts_match':True,
       'all_TRs_match':True,'all_beta_and_BOLD_affines_match':True,'all_external_local_events_hash_match':True,
       'runs_with_last_event_beyond_acquisition':sum(r['last_event_end_beyond_scan'] for r in results),
       'runs_with_valuation_beyond_acquisition':sum(r['valuation_events_ending_beyond_scan']>0 for r in results),
       'valuation_events_ending_beyond_acquisition':sum(r['valuation_events_ending_beyond_scan'] for r in results),
       'tail_after_last_valuation_seconds':{'min':min(r['scan_tail_after_last_valuation_seconds'] for r in results),
          'median':float(np.median([r['scan_tail_after_last_valuation_seconds'] for r in results]))},
       'seed':None,'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'script_sha256':sha(__file__),'python':platform.python_version(),'nibabel':nib.__version__,
       'limitations':'BOLD headers and identities checked, not a complete preprocessing audit or full BOLD volume hash.',
       'output_sha256':sha(out/'run_scan_identity.csv')}
    (out/'audit.json').write_text(json.dumps(a,indent=2)+'\n');shutil.copy2(__file__,out/'script_snapshot.py')
    print(json.dumps(a,indent=2))


if __name__=='__main__':main()
