"""Copy the research handoff and local inputs without changing source files.

Standard library only. Existing destinations must have identical bytes.
CSV preparation changes path columns only; scientific values remain strings.
"""
import csv
import hashlib
import io
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/path/to/external_data/minimal_lsa/2/outputs/food_valuation_vlm_handoff_20260915')
ARCHIVE = ROOT / 'archive/handoff_20260915'
EVENTS = Path('/path/to/external_data/fmriPrep_31out/events')
FOOD = Path('/path/to/local_data/FoodPics_extended_2022')
BEHAVIOR = 'analyses/behavior_agreement_nutrition_20260915/tables'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    records = []

    def copy(source, destination, kind):
        source_hash = digest(source)
        if destination.exists():
            if digest(destination) != source_hash:
                raise RuntimeError(f'Refusing to overwrite different file: {destination}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if digest(destination) != source_hash:
            raise RuntimeError(f'Copy hash mismatch: {destination}')
        records.append(dict(source=str(source), path=destination.relative_to(ROOT).as_posix(),
                            bytes=destination.stat().st_size, sha256=source_hash, kind=kind))

    original_manifest = json.loads((SOURCE / 'file_manifest.json').read_text())
    for entry in original_manifest['files']:
        p = SOURCE / entry['relative_path']
        if p.stat().st_size != entry['bytes'] or digest(p) != entry['sha256']:
            raise RuntimeError(f'Original manifest mismatch: {p}')
    for p in sorted(SOURCE.rglob('*')):
        if p.is_file() and not p.name.startswith('._') and p.name != '.DS_Store':
            copy(p, ARCHIVE / p.relative_to(SOURCE), 'handoff_snapshot')

    _, images = read_csv(ARCHIVE / 'data/image_files_568.csv')
    image_map = {}
    for row in images:
        source = Path(row['image_path'])
        destination = ROOT / 'data/raw/images' / source.name
        copy(source, destination, 'image')
        image_map[row['image_path']] = destination.relative_to(ROOT).as_posix()
    event_files = sorted(EVENTS.glob('sub-*/ses-*/func/*events.tsv'))
    event_map = {}
    for source in event_files:
        destination = ROOT / 'data/raw/events' / source.relative_to(EVENTS)
        copy(source, destination, 'events')
        event_map[str(source)] = destination.relative_to(ROOT).as_posix()
    copy(FOOD / 'food.pics.extended.database (Stand 2022).xlsx',
         ROOT / 'data/raw/metadata/food.pics.extended.database (Stand 2022).xlsx', 'metadata')
    code_root = Path('/path/to/external_data/A-densely-sampled-fMRI-dataset-for-investigating-food-valuation-main')
    for p in sorted(code_root.rglob('*')):
        if p.is_file() and not any(part.startswith('.') for part in p.relative_to(code_root).parts):
            copy(p, ROOT / 'archive/source_code/dataset_author' / p.relative_to(code_root), 'reference_code')
    copy(Path('/path/to/external_data/minimal_lsa/run_minimal_lsa.m'),
         ROOT / 'archive/source_code/run_minimal_lsa.m', 'reference_code')

    def prepare(relative_source, name, columns):
        source = ARCHIVE / relative_source
        fields, rows = read_csv(source)
        new_fields = fields + [f'{col}_source' for col in columns]
        for row in rows:
            for col, mapping in columns.items():
                row[f'{col}_source'] = row[col]
                row[col] = mapping[row[col]]
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)
        destination = ROOT / 'data/prepared' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = buffer.getvalue().encode('utf-8-sig')
        if destination.exists() and destination.read_bytes() != payload:
            raise RuntimeError(f'Refusing to overwrite changed prepared data: {destination}')
        destination.write_bytes(payload)
        records.append(dict(source=str(source.relative_to(ROOT)), path=destination.relative_to(ROOT).as_posix(),
                            bytes=len(payload), sha256=digest(destination), kind='prepared_paths',
                            modified_columns=list(columns), added_columns=[f'{c}_source' for c in columns]))

    prepare(f'{BEHAVIOR}/trial_ratings_with_nutrition.csv', 'trial_ratings_with_nutrition.csv',
            {'image_path': image_map, 'events_file': event_map})
    prepare(f'{BEHAVIOR}/image_agreement_nutrition.csv', 'image_agreement_nutrition.csv', {'image_path': image_map})
    prepare('data/image_files_568.csv', 'image_files_568.csv', {'image_path': image_map})
    copy(ARCHIVE / f'{BEHAVIOR}/subject_summary.csv', ROOT / 'data/prepared/subject_summary.csv', 'prepared_unchanged')
    copy(ARCHIVE / 'data/behavior_to_singletrial_beta_29subjects.csv',
         ROOT / 'data/external/behavior_to_singletrial_beta_29subjects.csv', 'brain_reference')

    locations = json.loads((ARCHIVE / 'data_locations.json').read_text())
    external_checks = {name: {'path': item['path'], 'exists': Path(item['path']).exists()}
                       for name, item in locations.items()}
    _, trials = read_csv(ROOT / 'data/prepared/trial_ratings_with_nutrition.csv')
    _, image_rows = read_csv(ROOT / 'data/prepared/image_agreement_nutrition.csv')
    _, beta_rows = read_csv(ROOT / 'data/external/behavior_to_singletrial_beta_29subjects.csv')
    key = lambda row: tuple(row[c] for c in ('subject_id', 'session_id', 'run_id', 'trial_id'))
    trial_index = {key(r): r for r in trials}
    beta_index = {key(r): r for r in beta_rows}
    assert len(trials) == len(trial_index) == len(beta_rows) == len(beta_index) == 15869
    assert trial_index.keys() == beta_index.keys()
    assert len({(r['subject_id'], r['image_id']) for r in trials}) == 15869
    subjects = {r['subject_id'] for r in trials}
    assert len(subjects) == 29 and not subjects.intersection({'sub-004', 'sub-010'})
    assert {r['rating'] for r in trials} == {'1', '2', '3', '4'}
    assert len(image_rows) == len(images) == 568
    assert {int(r['image_id']) for r in image_rows} == set(range(1, 569))
    for k, row in trial_index.items():
        beta = beta_index[k]
        assert row['image_id'] == beta['image_id'] and row['rating'] == beta['rating']
        assert float(row['onset']) == float(beta['onset_behavior']) == float(beta['onset_beta'])
    for col in ('image_path', 'events_file'):
        assert all((ROOT / p).is_file() for p in {r[col] for r in trials})
    beta_paths = {r['beta_file_local'] for r in beta_rows}
    spm_paths = {r['spm_mat_local'] for r in beta_rows}
    missing_beta = [p for p in sorted(beta_paths) if not Path(p).is_file()]
    missing_spm = [p for p in sorted(spm_paths) if not Path(p).is_file()]
    timestamp = datetime.now().astimezone().isoformat()
    report = dict(timestamp=timestamp, source=str(SOURCE), destination=str(ROOT),
                  original_manifest_entries_verified=len(original_manifest['files']),
                  file_counts=dict(Counter(r['kind'] for r in records)),
                  total_managed_bytes=sum(r['bytes'] for r in records),
                  copied_files_sha256_verified=True,
                  n_subjects=29, n_trials=len(trials), n_images=len(image_rows),
                  raw_event_subjects=len({p.relative_to(EVENTS).parts[0] for p in event_files}),
                  excluded_subjects=['sub-004', 'sub-010'],
                  behavior_beta_join='one_to_one_image_rating_onset_match',
                  beta_unique_paths=len(beta_paths), spm_unique_paths=len(spm_paths),
                  missing_beta_paths=missing_beta, missing_spm_paths=missing_spm,
                  registered_source_locations=external_checks,
                  brain_values_or_roi_quality_examined=False,
                  new_model_inference_or_analysis_run=False)
    save_json(ROOT / 'docs/migration/manifest_20260915.json', {'timestamp': timestamp, 'files': records})
    save_json(ROOT / 'docs/migration/validation_20260915.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'registered_source_locations'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
