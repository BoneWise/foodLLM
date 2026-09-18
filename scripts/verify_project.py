"""Read-only verification of migrated local inputs; optional external checks."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'docs/migration/manifest_20260915.json').read_text())
    errors = []
    for record in manifest['files']:
        path = ROOT / record['path']
        if not path.is_file():
            errors.append(f'Missing: {record["path"]}')
            continue
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != record['sha256'] or path.stat().st_size != record['bytes']:
            errors.append(f'Hash/size mismatch: {record["path"]}')
        if record['kind'] == 'prepared_paths':
            original = rows(ROOT / record['source'])
            prepared = rows(path)
            if len(original) != len(prepared):
                errors.append(f'Row count changed: {record["path"]}')
            for old, new in zip(original, prepared):
                for col, value in old.items():
                    actual_value = new.get(f'{col}_source') if col in record['modified_columns'] else new.get(col)
                    if actual_value != value:
                        errors.append(f'Source value changed: {record["path"]}: {col}')
                        break
                for col in record['modified_columns']:
                    if not (ROOT / new[col]).is_file():
                        errors.append(f'Local reference missing: {new[col]}')
    config = json.loads((ROOT / 'configs/project.json').read_text())
    for name, value in config['paths'].items():
        if not (ROOT / value).exists():
            errors.append(f'Configured local path missing: {name}')
    trials = rows(ROOT / config['paths']['trial_ratings'])
    images = rows(ROOT / config['paths']['image_summary'])
    sample = config['confirmed_behavior_sample']
    subjects = {r['subject_id'] for r in trials}
    image_counts = Counter(r['image_id'] for r in trials)
    sample_ok = (len(trials) == sample['trials'] and len(subjects) == sample['subjects']
                 and len(images) == len(image_counts) == sample['images']
                 and not subjects.intersection(sample['excluded_subjects'])
                 and len({(r['subject_id'], r['image_id']) for r in trials}) == len(trials))
    if not sample_ok:
        errors.append('Behavior sample differs from confirmed baseline')
    for image in images:
        if int(image['n']) != image_counts[image['image_id']]:
            errors.append(f'Image rating count mismatch: {image["image_id"]}')
    external_count = 0
    if args.external:
        for path in config['external'].values():
            if not Path(path).exists():
                errors.append(f'External root missing: {path}')
        beta = rows(ROOT / config['paths']['behavior_beta_mapping'])
        paths = {r[col] for r in beta for col in ('beta_file_local', 'spm_mat_local')}
        external_count = len(paths)
        errors.extend(f'External file missing: {p}' for p in sorted(paths) if not Path(p).is_file())
    print(json.dumps({'ok': not errors, 'managed_files_checked': len(manifest['files']),
                      'subjects': len(subjects), 'trials': len(trials), 'images': len(images),
                      'external_files_checked': external_count, 'errors': errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
