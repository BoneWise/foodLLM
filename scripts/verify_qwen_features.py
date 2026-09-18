"""Validate saved model features and provenance without loading human outcomes."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def key(row):
    return (row['prompt_id'], row['image_id'], row['condition'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--pilot', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve historical verification; choose a new output file')
    root = Path(args.features)
    audit = json.loads((root/'audit.json').read_text())
    rows = json.loads((root/'records.json').read_text())
    expected = {key(r): r for r in json.loads(Path(args.manifest).read_text())['inputs']}
    pilot = Path(args.pilot)
    pilot_audit = json.loads((pilot/'audit.json').read_text())
    assert audit['status'] == 'complete' and audit['mode'] == 'extract'
    assert len(rows) == len(expected) == audit['inputs'] == 7202
    assert len({key(r) for r in rows}) == len(rows)
    assert {key(r) for r in rows} == set(expected)
    assert audit['input_manifest_sha256'] == digest(args.manifest)
    assert audit['model_files'] == pilot_audit['model_files']
    assert audit['prompt_config_sha256'] == pilot_audit['prompt_config_sha256']
    assert audit['versions'] == pilot_audit['versions']
    assert len(audit['pilot_reference_exact_matches']) == pilot_audit['inputs'] == 30
    assert audit['pilot_reference_audit_sha256'] == digest(pilot/'audit.json')
    with (root/'records.jsonl').open() as f:
        streamed = [json.loads(line) for line in f]
    assert streamed == rows
    counts = Counter()
    quality = defaultdict(list)
    pilot_matches = []
    for index, r in enumerate(rows):
        k = key(r)
        expected_row = expected[k]
        for field in ('image_sha256', 'true_kcal_100g', 'presented_kcal_100g', 'donor_image_id', 'primary_eligible'):
            assert r[field] == expected_row[field], (k, field)
        cache = root/r['cache_path']
        assert digest(cache) == r['cache_sha256'], k
        rendered = root/('_'.join(k)+'_rendered.txt')
        assert digest(rendered) == r['prompt_sha256'], k
        with np.load(cache, allow_pickle=False) as z:
            assert z['hidden'].shape == (32, 4096) and z['final_norm'].shape == (4096,)
            assert np.isfinite(z['hidden']).all() and np.isfinite(z['final_norm']).all()
            lp = z['candidate_logprobs'].astype(float)
            probs = z['rating_probs'].astype(float)
            assert lp.shape == probs.shape == (4,)
            assert np.isfinite(lp).all() and np.isfinite(probs).all()
            assert np.allclose(probs.sum(), 1., atol=1e-6)
            normalized = np.exp(lp-lp.max()); normalized /= normalized.sum()
            assert np.allclose(probs, normalized, atol=1e-6)
            assert np.array_equal(probs, np.asarray(r['rating_probs']))
            assert np.array_equal(lp, np.asarray(r['candidate_logprobs']))
            assert np.isclose(np.exp(lp).sum(), r['valid_answer_mass'], atol=1e-6)
            assert np.isclose(probs @ np.arange(1,5), r['expected_rating'], atol=1e-6)
            assert int(probs.argmax()+1) == r['hard_rating']
            assert hashlib.sha256(z['input_ids'].tobytes()).hexdigest() == r['input_ids_sha256']
            assert z['input_ids'].shape == (1, r['token_count'])
            assert int(z['input_ids'][0,-1]) == r['readout_token_id']
            assert r['readout_index'] == r['token_count']-1
            ref = pilot/cache.name
            if ref.exists():
                with np.load(ref, allow_pickle=False) as old:
                    assert all(np.array_equal(z[name], old[name]) for name in z.files)
                pilot_matches.append('_'.join(k))
        assert r['candidate_token_ids'] == [16,17,18,19]
        assert r['enable_thinking'] is False
        assert r['norm_error'] == r['lm_head_error'] == 0
        assert r['nutrition_token_ids'] if r['condition'] != 'V' else r['nutrition_token_ids'] == []
        counts[(r['prompt_id'],r['condition'])] += 1
        quality[(r['prompt_id'],r['condition'])].append(r)
        if (index+1) % 1000 == 0:
            print(json.dumps({'verified':index+1,'total':len(rows)}), flush=True)
    assert set(pilot_matches) == set(audit['pilot_reference_exact_matches'])
    assert len(pilot_matches) == 30
    assert len(list(root.glob('*.npz'))) == len(rows)
    summary=[]
    for (prompt, condition), group in sorted(quality.items()):
        expected_n = 568 if condition in ('V','T') else 493
        assert len(group) == expected_n
        summary.append({'prompt':prompt,'condition':condition,'inputs':len(group),
            'valid_mass_min':min(r['valid_answer_mass'] for r in group),
            'valid_mass_max':max(r['valid_answer_mass'] for r in group),
            'global_argmax_not_rating':sum(not r['global_argmax_is_rating'] for r in group),
            'token_count_min':min(r['token_count'] for r in group),
            'token_count_max':max(r['token_count'] for r in group)})
    report={'status':'passed','verified_inputs':len(rows),'pilot_exact_matches':len(pilot_matches),
        'all_cache_hashes_shapes_finiteness_and_probabilities_verified':True,
        'full_key_coverage_and_manifest_agreement':True,'quality_by_prompt_condition':summary,
        'no_output_based_exclusions':True,'human_or_brain_targets_loaded':False,
        'provenance':{'script_sha256':digest(__file__),'input_manifest_sha256':digest(args.manifest),
                      'extraction_audit_sha256':digest(root/'audit.json'),
                      'pilot_audit_sha256':digest(pilot/'audit.json'),'numpy_version':np.__version__}}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','verified_inputs':len(rows),'pilot_exact_matches':len(pilot_matches)}))


if __name__ == '__main__':
    main()
