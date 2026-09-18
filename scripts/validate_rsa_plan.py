"""Validate inherited RSA inputs without computing alignment or changing data."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New project-relative output directory')
    parser.add_argument('--config', help='Project-relative RSA config; defaults to current project entry')
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT / 'outputs') or out.exists():
        raise ValueError('Use a new directory inside project outputs; history is not overwritten')
    project = json.loads((ROOT / 'configs/project.json').read_text())
    config_path = ROOT / (args.config or project['current_analysis_config'])
    cfg = json.loads(config_path.read_text())
    hashes = {}

    def record(path):
        path = Path(path)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def rows(path):
        record(path)
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    for name in cfg['inputs'].values():
        assert (ROOT / name).exists(), name
    idx_dir = ROOT / cfg['inputs']['analysis_index']
    idx = json.loads((idx_dir / 'index.json').read_text())
    record(idx_dir / 'index.json')
    arrays = {}
    for name in ['human_ratings', 'roi_beta', 'primary_mask', 'session_ids', 'run_ids', 'content_covariates']:
        path = idx_dir / (name + '.npy')
        record(path)
        arrays[name] = np.load(path, allow_pickle=False)
    primary = arrays['primary_mask']
    images = idx['image_ids']
    subjects = idx['subject_ids']
    assert all(isinstance(x, str) for x in images + subjects)
    assert len(set(images)) == len(images) == 568
    assert len(set(subjects)) == len(subjects) == 29
    assert primary.dtype == bool and primary.sum() == 493
    ratings, brain = arrays['human_ratings'], arrays['roi_beta']
    assert ratings.shape == brain.shape == (29, 568)
    assert np.isfinite(ratings).sum() == 15869
    assert np.isfinite(brain).sum() == 15867
    assert np.isfinite(ratings[:, primary]).sum() == 13767
    assert np.isfinite(brain[:, primary]).sum() == 13765
    assert idx['content_columns'][-1] == 'correct_kcal_100g'
    assert np.isfinite(arrays['content_covariates'][primary, -1]).all()
    wanted = {x for x, keep in zip(images, primary) if keep}
    matrices = []
    for prompt in ['A', 'B']:
        for condition in cfg['model']['conditions']:
            directory = ROOT / cfg['inputs']['model_matrices'] / f'{prompt}_{condition}'
            table = rows(directory / 'image_index.csv')
            ids = [r['image_id'] for r in table]
            assert len(ids) == len(set(ids)) and wanted <= set(ids)
            assert [int(r['matrix_row']) for r in table] == list(range(len(table)))
            assert all(r['prompt_id'] == prompt and r['condition'] == condition for r in table)
            matrix = np.load(directory / 'hidden.npy', mmap_mode='r', allow_pickle=False)
            assert matrix.shape == (32, len(ids), 4096)
            matrices.append({'prompt': prompt, 'condition': condition, 'shape': list(matrix.shape),
                             'dtype': str(matrix.dtype), 'all_primary_images_present': True})
    key_names = ['subject_id', 'session_id', 'run_id', 'trial_id', 'image_id']
    key = lambda r: tuple(r[n] for n in key_names)
    mapping_rows = rows(ROOT / cfg['inputs']['beta_mapping'])
    mapping = {key(r): r for r in mapping_rows}
    assert len(mapping) == len(mapping_rows) == 15869
    eligibility = rows(ROOT / cfg['inputs']['brain_eligibility'])
    assert len({key(r) for r in eligibility}) == len(eligibility) == 15869
    selected = [r for r in eligibility if r['neural_analysis_eligible'].lower() == 'true' and r['image_id'] in wanted]
    assert len(selected) == 13765
    observed = set()
    external_missing = []
    subject_index = {s: i for i, s in enumerate(subjects)}
    image_index = {s: i for i, s in enumerate(images)}
    for r in selected:
        k = key(r)
        assert k in mapping
        si, ii = subject_index[r['subject_id']], image_index[r['image_id']]
        assert (si, ii) not in observed
        observed.add((si, ii))
        assert arrays['session_ids'][si, ii] == r['session_id']
        assert arrays['run_ids'][si, ii] == r['session_id'] + '/' + r['run_id']
        assert np.isclose(brain[si, ii], float(r['roi_beta_mean']), rtol=0, atol=1e-12)
        if not Path(mapping[k]['beta_file_local']).is_file():
            external_missing.append(k)
    masks = []
    for subject in subjects:
        path = ROOT / cfg['inputs']['fixed_roi_support'] / f'{subject}_fixed_support.nii.gz'
        record(path)
        obj = nib.load(path)
        mask = obj.get_fdata()
        assert np.isfinite(mask).all() and set(np.unique(mask)) <= {0, 1}
        count = int(np.count_nonzero(mask))
        expected = {int(r['fixed_support_voxels']) for r in selected if r['subject_id'] == subject}
        assert expected == {count} and count > 1
        masks.append({'subject_id': subject, 'voxels': count, 'shape': list(mask.shape)})
    for name in ['protocol']:
        record(ROOT / cfg[name])
    record(config_path)
    record(Path(__file__))
    for name in ['roi_config', 'trial_exclusions', 'prompt_config', 'stimulus_manifest', 'permutations']:
        record(ROOT / cfg['inputs'][name])
    assert cfg['rdm']['train_test_split'] is None
    assert cfg['neural_inference']['unit'] == 'participant'
    assert cfg['neural_inference']['contrasts'] == ['T-V', 'T-mean_P_RSA']
    if cfg['version'] == 'v06.1':
        assert cfg['neural_inference']['run_in_first_pass'] is False
        assert cfg['neural_inference']['method'] is None
        assert cfg['neural_inference']['resampling_count'] is None
        assert cfg['behavior_anchor']['bootstrap_enabled'] is False
        assert cfg['reserved_sensitivity']['run_in_first_pass'] is False
        assert cfg['acquisition_QC']['run_identity'] == ['session_id', 'run_id']
        assert cfg['acquisition_QC']['automatic_preprocessing'] is False
        assert cfg['model']['execution_order'] == ['A', 'B_after_A_pipeline_validation']
    else:
        assert cfg['neural_inference']['family_size'] == 2 * len(cfg['model']['layers']) == 64
    report = {
        'status': 'input_inventory_passed' if not external_missing else 'local_inventory_passed_external_beta_missing',
        'real_RSA_effects_computed': False, 'multivoxel_patterns_extracted': False,
        'protocol_version': cfg['version'],
        'unsmoothed_source_reaudited': False,
        'inference_implementation_validated': False, 'post_result_follow_up': True,
        'subjects': 29, 'primary_images': 493, 'primary_brain_trials': len(selected),
        'primary_behavior_ratings': 13767,
        'per_subject_images_range': [int(x) for x in (np.isfinite(brain[:, primary]).sum(1).min(), np.isfinite(brain[:, primary]).sum(1).max())],
        'per_image_subjects_range': [int(x) for x in (np.isfinite(brain[:, primary]).sum(0).min(), np.isfinite(brain[:, primary]).sum(0).max())],
        'external_primary_beta_paths_checked': len(selected), 'external_missing': external_missing,
        'model_matrices': matrices, 'fixed_masks': masks,
        'scope': 'Index, eligibility, mean ledger, matrix headers and ROI masks only; no beta voxel values or full hidden matrix hashes rechecked.',
        'previous_feature_validation': 'outputs/20260916_Qwen_AB分析矩阵_v01',
        'input_sha256': hashes,
        'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'nibabel': nib.__version__},
        'parameters_and_seeds': 'config_snapshot.json',
    }
    out.mkdir(parents=True)
    (out / 'plan_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(config_path, out / 'config_snapshot.json')
    shutil.copy2(ROOT / cfg['protocol'], out / 'protocol_snapshot.md')
    shutil.copy2(__file__, out / 'script_snapshot.py')
    print(json.dumps({k: report[k] for k in ['status', 'subjects', 'primary_images', 'primary_brain_trials', 'per_subject_images_range', 'per_image_subjects_range', 'external_primary_beta_paths_checked']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
