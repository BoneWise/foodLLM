"""Extract an independent vmPFC ROI and audit existing SPM designs, without effect tests."""
import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def roi_values(path, reference, slices, selection):
    img = nib.load(path)
    if img.shape != reference.shape or not np.allclose(img.affine, reference.affine):
        raise ValueError(f'Geometry differs: {path}')
    return np.asarray(img.dataobj[slices], dtype=np.float32)[selection]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists() or not out.resolve().is_relative_to(ROOT / 'outputs'):
        parser.error('Use a new directory inside outputs')
    out.mkdir(parents=True)
    cfg_path = ROOT / 'configs/roi_preparation_v01.json'
    cfg = json.loads(cfg_path.read_text())
    mapping_path = ROOT / 'data/external/behavior_to_singletrial_beta_29subjects.csv'
    with mapping_path.open(encoding='utf-8-sig') as f:
        mapping = list(csv.DictReader(f))
    runs = defaultdict(list)
    for r in mapping:
        runs[(r['subject_id'], r['session_id'], r['run_id'])].append(r)
    ref = nib.load(mapping[0]['beta_file_local'])
    ijk = np.indices(ref.shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(ref.affine, ijk)
    sphere = (np.linalg.norm(xyz - np.array(cfg['center_mm']), axis=1) <= cfg['radius_mm']).reshape(ref.shape)
    vox = np.where(sphere)
    slices = tuple(slice(int(v.min()), int(v.max()) + 1) for v in vox)
    local_sphere = sphere[slices]
    n_sphere = int(sphere.sum())
    nib.save(nib.Nifti1Image(sphere.astype(np.uint8), ref.affine), out / 'vmPFC_sphere.nii.gz')
    subjects = sorted({r['subject_id'] for r in mapping})
    supports = {s: np.ones(n_sphere, dtype=bool) for s in subjects}
    run_records = []
    design_errors = []
    input_records = []
    for idx, (run, rows) in enumerate(sorted(runs.items())):
        directory = Path(rows[0]['spm_mat_local']).parent
        mask_path = directory / 'mask.nii'
        support = roi_values(mask_path, ref, slices, local_sphere) > 0
        supports[run[0]] &= support
        spm_path = Path(rows[0]['spm_mat_local'])
        spm = loadmat(spm_path, simplify_cells=True)['SPM']
        names = np.atleast_1d(spm['xX']['name']).tolist()
        sess = spm['Sess']
        conditions = sess['U'] if isinstance(sess['U'], list) else np.atleast_1d(sess['U']).tolist()
        if isinstance(sess['U'], dict):
            conditions = [sess['U']]
        u_by_name = {str(np.atleast_1d(u['name'])[0]): u for u in conditions}
        errors = []
        for r in rows:
            beta_index = int(re.search(r'beta_(\d+)', Path(r['beta_file_local']).name)[1]) - 1
            target = f"target_trial_{int(r['trial_id']):04d}"
            if names[beta_index] != f'Sn(1) {target}*bf(1)':
                errors.append(f'{r["trial_id"]}: beta column mismatch')
            u = u_by_name.get(target)
            if u is None or abs(float(np.atleast_1d(u['ons'])[0]) - float(r['onset_beta'])) > 1e-6:
                errors.append(f'{r["trial_id"]}: onset mismatch')
            if u is not None and u.get('P', {}).get('name') != 'none':
                errors.append(f'{r["trial_id"]}: unexpected target pmod')
        resms = roi_values(directory / 'ResMS.nii', ref, slices, local_sphere)
        rec = {'subject_id': run[0], 'session_id': run[1], 'run_id': run[2],
               'valid_behavior_trials': len(rows), 'sphere_voxels': n_sphere,
               'mask_covered_voxels': int(support.sum()), 'mask_coverage': float(support.mean()),
               'design_columns': len(names), 'design_rows': int(np.asarray(spm['xX']['X']).shape[0]),
               'effective_residual_df': float(spm['xX']['erdf']),
               'resms_median_in_run_roi': float(np.median(resms[support])) if support.any() else None,
               'design_mapping_ok': not errors}
        run_records.append(rec)
        design_errors.extend({'run': '/'.join(run), 'error': e} for e in errors)
        input_records.append({'spm_path': str(spm_path), 'spm_sha256': sha(spm_path),
                              'mask_path': str(mask_path), 'mask_sha256': sha(mask_path)})
        if idx % 20 == 0:
            print(json.dumps({'phase': 'mask_and_design', 'runs_done': idx + 1, 'runs_total': len(runs)}), flush=True)
    subject_records = []
    for s, support in supports.items():
        whole = np.zeros(ref.shape, dtype=np.uint8)
        whole[vox] = support
        nib.save(nib.Nifti1Image(whole, ref.affine), out / f'{s}_fixed_support.nii.gz')
        subject_records.append({'subject_id': s, 'fixed_support_voxels': int(support.sum()),
                                'sphere_voxels': n_sphere, 'coverage': float(support.mean()),
                                'geometry_eligible': float(support.mean()) >= cfg['minimum_sphere_coverage']})
    eligible = {r['subject_id']: r['geometry_eligible'] for r in subject_records}
    trial_records = []
    payload_hash = hashlib.sha256()
    for idx, r in enumerate(mapping):
        vals = roi_values(r['beta_file_local'], ref, slices, local_sphere)
        selected = vals[supports[r['subject_id']]]
        finite = np.isfinite(selected)
        finite_fraction = float(finite.mean()) if selected.size else 0.
        payload_hash.update(r['beta_file_local'].encode())
        payload_hash.update(vals.tobytes())
        rec = {k: r[k] for k in ('subject_id', 'session_id', 'run_id', 'trial_id', 'image_id')}
        rec.update({'roi_beta_mean': float(selected.mean()) if selected.size and finite.all() else None,
                    'fixed_support_voxels': int(selected.size), 'finite_fraction': finite_fraction,
                    'geometry_and_finite_eligible': eligible[r['subject_id']] and finite_fraction >= cfg['minimum_finite_fraction_per_trial'],
                    'legacy_quality_label_not_applied': r['quality_label']})
        trial_records.append(rec)
        if idx % 1000 == 0:
            print(json.dumps({'phase': 'roi_beta', 'trials_done': idx + 1, 'trials_total': len(mapping)}), flush=True)
    for rec in subject_records:
        rows = [r for r in trial_records if r['subject_id'] == rec['subject_id']]
        vals = np.array([r['roi_beta_mean'] for r in rows if r['roi_beta_mean'] is not None])
        rec['valid_finite_trials'] = int(len(vals))
        rec['roi_beta_sd_descriptive'] = float(vals.std()) if len(vals) else None
        rec['eligible_trials'] = sum(r['geometry_and_finite_eligible'] for r in rows)
    write_csv(out / 'run_quality.csv', run_records)
    write_csv(out / 'subject_quality.csv', subject_records)
    write_csv(out / 'roi_trial_responses.csv', trial_records)
    summary = {'created_at': datetime.now().astimezone().isoformat(), 'rules': cfg,
               'subjects': len(subjects), 'runs': len(runs), 'trials': len(mapping),
               'sphere_voxels': n_sphere, 'eligible_subjects_geometry': sum(eligible.values()),
               'eligible_trials_geometry_and_finite': sum(r['geometry_and_finite_eligible'] for r in trial_records),
               'design_errors': design_errors, 'subject_records': subject_records,
               'limitations': ['No human/model correlation or effect test', 'Geometry/finite checks do not establish test-retest reliability',
                              'ROI-payload hash, not full external beta-volume hashes', 'No same-person same-image repeats; power remains unmeasured'],
               'provenance': {'mapping_sha256': sha(mapping_path), 'rules_sha256': sha(cfg_path),
                              'script_sha256': sha(__file__), 'beta_sphere_payload_sha256': payload_hash.hexdigest(),
                              'git_base': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'nibabel': nib.__version__, 'scipy': scipy.__version__},
                              'random_seed': None, 'run_input_hashes': input_records}}
    (out / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('subjects','runs','trials','sphere_voxels','eligible_subjects_geometry','eligible_trials_geometry_and_finite','design_errors')}), flush=True)
    if design_errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
