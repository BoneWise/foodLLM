"""Read-only verification of existing caches for the HCP design; no new RSA."""
import ast
import hashlib
import json
import platform
import struct
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def npy_shape(path):
    with path.open('rb') as f:
        assert f.read(6) == b'\x93NUMPY'
        major, _ = f.read(2)
        size = struct.unpack('<H' if major == 1 else '<I', f.read(2 if major == 1 else 4))[0]
        return ast.literal_eval(f.read(size).decode('utf-8'))['shape']


def main():
    out = ROOT / 'outputs/20260917_HCP设计对齐核查_v01'
    assert not out.exists(), 'Preserve previous audit; use a new version for reruns.'
    cfgpath = ROOT / 'configs/hcp_regional_alignment_v08_draft.json'
    cfg = json.loads(cfgpath.read_text())
    labels = [r for group in cfg['HCP_groups'].values() for r in group]
    assert len(labels) == len(set(labels)) == 18
    assert 2 * len(labels) == cfg['HCP_unilateral_parcels'] == 36
    assert len(cfg['HCP_groups']['lateral_OFC'] + cfg['HCP_groups']['medial_OFC']) == 9
    hashes = {str(cfgpath.relative_to(ROOT)): digest(cfgpath)}
    oldcfgpath = ROOT / cfg['source_execution']
    oldcfg = json.loads(oldcfgpath.read_text())
    receiptpath = ROOT / 'outputs/20260917_四营养RSA报告_v01/completion_receipt.json'
    receipt = json.loads(receiptpath.read_text())
    assert digest(oldcfgpath) == receipt['execution_config_unchanged_sha256']
    for p in [oldcfgpath, receiptpath]:
        hashes[str(p.relative_to(ROOT))] = digest(p)
    for name, expected in oldcfg['manifest_sha256'].items():
        p = ROOT / oldcfg['metadata_preparation'] / name
        assert digest(p) == expected
        hashes[str(p.relative_to(ROOT))] = expected
    indices, identities, report = {}, {}, {}
    for prompt in ['A', 'B']:
        source = ROOT / cfg['legacy_references'][prompt + '_cache']
        audit = json.loads((source / 'audit.json').read_text())
        assert audit['status'] == 'complete'
        assert audit['all_conditions_attributes_ROIs_same_pair_masks'] is True
        hashes[str((source / 'audit.json').relative_to(ROOT))] = digest(source / 'audit.json')

        def verify(rel):
            p = source / rel
            d = digest(p)
            assert d == audit['output_sha256'][rel], str(p)
            hashes[str(p.relative_to(ROOT))] = d
            return p, d

        ip, _ = verify('index.json')
        ix = json.loads(ip.read_text())
        indices[prompt] = ix
        assert len(ix['image_ids']) == len(set(ix['image_ids'])) == 401
        assert len(ix['subject_ids']) == 29
        assert ix['ROIs'] == ['lOFC', 'mOFC', 'vmPFC']
        assert ix['conditions'] == cfg['conditions']
        assert ix['nutrients'] == cfg['nutrients']
        assert ix['layers'] == list(range(1, 33))
        roi_counts = {}
        for roi in ix['ROIs']:
            sizes = []
            for sub in ix['subject_ids']:
                pairpath, pairhash = verify(f'{roi}/brain/{sub}/global_image_rows.npy')
                n = npy_shape(pairpath)[0]
                path, rdmhash = verify(f'{roi}/brain/{sub}/RDM.npy')
                assert npy_shape(path) == (n, n)
                identities[prompt, roi, sub] = (pairhash, rdmhash)
                sizes.append(n)
            assert sum(sizes) == 11207
            roi_counts[roi] = {'participants': len(sizes), 'trial_sum': sum(sizes), 'min_images': min(sizes), 'max_images': max(sizes)}
        for sub in ix['subject_ids']:
            assert len({identities[prompt, roi, sub][0] for roi in ix['ROIs']}) == 1
        for nutrient in ix['nutrients']:
            p, _ = verify(nutrient + '/neural_scores.npy')
            assert npy_shape(p) == (3, 29, 32, 7)
            p, _ = verify(nutrient + '/contrasts.npy')
            assert npy_shape(p) == (3, 29, 32, 2)
        report[prompt] = roi_counts
    for key in ['image_ids', 'subject_ids', 'ROIs', 'nutrients', 'conditions', 'layers']:
        assert indices['A'][key] == indices['B'][key]
    for roi in indices['A']['ROIs']:
        for sub in indices['A']['subject_ids']:
            assert identities['A', roi, sub] == identities['B', roi, sub]
    result = {
        'status': 'passed_design_and_legacy_cache_integrity',
        'created': datetime.now().astimezone().isoformat(),
        'HCP_bilateral_pairs': 18, 'HCP_unilateral_parcels': 36,
        'legacy_ROIs': 3, 'final_ROI_series': 21,
        'legacy_cache': report,
        'AB_same_brain_RDM_and_index_hashes': True,
        'cache_reuse': 'eligible_for_existing_pair_sets; HCP comparison pair-set equivalence not yet testable',
        'HCP_atlas_checked': False, 'HCP_coverage_checked': False,
        'new_RSA_computed': False, 'inference_run': False,
        'limitations': ['Hash integrity and headers checked; not a fresh numerical RSA recomputation.', 'No complete model-hidden-array rehash in this design audit.', 'ROI reliability not established.'],
        'external_beta_root_exists': (ROOT / 'configs/project.json').exists() and Path('/path/to/external_data/minimal_lsa/output/models').exists(),
        'input_sha256': hashes,
        'script_sha256': digest(Path(__file__)),
        'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'seed': None, 'dependencies': {'python': platform.python_version(), 'packages': 'standard_library_only'}
    }
    out.mkdir(parents=True)
    (out / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (out / 'script_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: result[k] for k in ['status', 'final_ROI_series', 'legacy_cache', 'HCP_coverage_checked']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
