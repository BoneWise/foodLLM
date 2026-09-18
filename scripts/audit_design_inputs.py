"""Read-only structural audit for the nutrient-assisted prediction design.

Writes a new audit directory only. Does not clean ratings, select participants,
fit behavioral models, construct final permutations or estimate brain responses.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def describe(values):
    values = sorted(values)
    n = len(values)
    return {"n": n, "min": values[0] if n else None,
            "max": values[-1] if n else None,
            "median": (values[(n-1)//2] + values[n//2])/2 if n else None}


def nifti_header(path):
    with path.open("rb") as stream:
        raw = stream.read(348)
    endian = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    if len(raw) != 348 or struct.unpack(endian + "i", raw[:4])[0] != 348:
        raise ValueError("Not an uncompressed NIfTI-1 header")
    dims = struct.unpack_from(endian + "8h", raw, 40)
    return {"shape": list(dims[1:1+dims[0]]),
            "voxel_sizes": list(struct.unpack_from(endian + "8f", raw, 76)[1:4]),
            "datatype": struct.unpack_from(endian + "h", raw, 70)[0],
            "qform_code": struct.unpack_from(endian + "h", raw, 252)[0],
            "sform_code": struct.unpack_from(endian + "h", raw, 254)[0],
            "sform_rows": [list(struct.unpack_from(endian + "4f", raw, p))
                           for p in (280, 296, 312)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        parser.error("Output must be inside project outputs")
    if output.exists():
        parser.error("Use a new output version; existing results are preserved")
    config_path = ROOT / "configs/project.json"
    config = json.loads(config_path.read_text())
    paths = {k: ROOT / config["paths"][k] for k in
             ("trial_ratings", "image_summary", "image_index", "behavior_beta_mapping")}
    tables = {k: read_rows(p) for k, p in paths.items()}
    trials, images, index, mapping = (tables[k] for k in paths)
    errors = []
    key = lambda r: tuple(r[k] for k in ("subject_id", "session_id", "run_id", "trial_id"))
    by_trial = {key(r): r for r in trials}
    by_image = {r["image_id"]: r for r in images}
    by_index = {r["image_id"]: r for r in index}
    if len(by_trial) != len(trials):
        errors.append("Repeated subject/session/run/trial keys")
    if len({key(r) for r in mapping}) != len(mapping):
        errors.append("Repeated beta mapping keys")
    if set(by_trial) != {key(r) for r in mapping}:
        errors.append("Behavior and beta mapping keys differ")
    if set(by_image) != set(by_index):
        errors.append("Image summary and image index IDs differ")
    rating_values = Counter(r["rating"] for r in trials)
    subject_images = Counter(r["subject_id"] for r in trials)
    image_counts = Counter(r["image_id"] for r in trials)
    image_ratings = defaultdict(list)
    run_counts = Counter(tuple(r[k] for k in ("subject_id", "session_id", "run_id")) for r in trials)
    sessions = Counter()
    for subject, session, _ in run_counts:
        sessions[(subject, session)] += 1
    for r in trials:
        image_ratings[r["image_id"]].append(float(r["rating"]))
        ir = by_image[r["image_id"]]
        for field in ("kcal_100g", "fat_100g", "grams_total", "category_name"):
            if r[field] != ir[field]:
                errors.append(f"Trial/image metadata differs: {field}, {r['image_id']}")
    for image_id, r in by_image.items():
        vals = image_ratings[image_id]
        if int(r["n"]) != len(vals) or abs(float(r["mean_rating"])-sum(vals)/len(vals)) > 1e-12:
            errors.append(f"Stored mean/count inconsistent: {image_id}")
    for r in mapping:
        t = by_trial.get(key(r))
        if not t:
            continue
        if (r["image_id"] != t["image_id"] or r["rating"] != t["rating"]
                or r["rating_value"] != t["rating"]
                or abs(float(r["onset_beta"])-float(t["onset"])) > 1e-8):
            errors.append(f"Behavior/beta identity or onset mismatch: {key(r)}")

    fields = ("kcal_100g", "fat_100g", "protein_100g", "carbs_100g", "fibre_100g",
              "grams_total", "kcal_total", "red", "green", "blue", "object_size",
              "intensity_norm", "contrast")
    numeric = {}
    for field in fields:
        good = [float(r[field]) for r in images if finite(r[field])]
        numeric[field] = {**describe(good), "missing_ids": [r["image_id"] for r in images if not finite(r[field])],
                          "unique_values": len(set(good)), "zero_count": sum(x == 0 for x in good)}
    energy_errors = [abs(float(r["kcal_total"]) - float(r["kcal_100g"])*float(r["grams_total"])/100)
                     for r in images]
    groups = defaultdict(list)
    for r in images:
        groups[(r["category_name"], r["processing"])].append(r)
    group_rows, zero_candidates, candidate_counts = [], [], []
    for (category, processing), group in sorted(groups.items()):
        vals = [float(r["kcal_100g"]) for r in group]
        group_rows.append({"category": category, "processing": processing, **describe(vals),
                           "distinct_display_values_1dp": len({round(x, 1) for x in vals})})
        for r in group:
            n = sum(round(float(q["kcal_100g"]), 1) != round(float(r["kcal_100g"]), 1)
                    for q in group if q["image_id"] != r["image_id"])
            candidate_counts.append(n)
            if n == 0:
                zero_candidates.append({k: r[k] for k in ("image_id", "food_name", "category_name", "processing", "kcal_100g")})
    named_groups = defaultdict(list)
    for r in images:
        named_groups[r["food_name"].strip().casefold()].append(r["image_id"])
    duplicate_names = {k: v for k, v in named_groups.items() if len(v) > 1}
    image_hash_groups = defaultdict(list)
    for r in index:
        p = ROOT / r["image_path"]
        if not p.is_file():
            errors.append(f"Image file missing: {p}")
        else:
            image_hash_groups[digest(p)].append(r["image_id"])

    external_paths = {r[col] for r in mapping for col in ("beta_file_local", "spm_mat_local")}
    external_missing = [p for p in sorted(external_paths) if not Path(p).is_file()]
    run_representatives = {}
    for r in mapping:
        run_representatives.setdefault(tuple(r[k] for k in ("subject_id", "session_id", "run_id")), r)
    geometries, header_errors = defaultdict(list), []
    for run, r in sorted(run_representatives.items()):
        p = Path(r["beta_file_local"])
        if not p.is_file():
            continue
        try:
            h = nifti_header(p)
            geometries[json.dumps(h, sort_keys=True)].append(list(run))
        except (ValueError, OSError, struct.error) as exc:
            header_errors.append({"path": str(p), "error": str(exc)})
    sample = config["confirmed_behavior_sample"]
    if (len(trials), len(subject_images), len(images)) != (sample["trials"], sample["subjects"], sample["images"]):
        errors.append("Configured sample differs")
    if set(subject_images).intersection(sample["excluded_subjects"]):
        errors.append("Excluded subjects present")
    repeats = len(trials)-len({(r["subject_id"], r["image_id"]) for r in trials})
    if repeats:
        errors.append("Repeated within-subject images present")

    verifier = subprocess.run([sys.executable,
                               str(ROOT / "scripts/verify_project.py"), "--external"],
                              capture_output=True, text=True, check=False, cwd=ROOT)
    verification = json.loads(verifier.stdout)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "Read-only design compatibility audit; no hypothesis tests, participant selection or ROI response extraction",
        "behavior": {"subjects": len(subject_images), "images": len(images), "trials": len(trials),
                     "rating_counts": dict(rating_values), "within_subject_image_repeats": repeats,
                     "ratings_per_image": describe(list(image_counts.values())),
                     "images_per_subject": dict(sorted(subject_images.items())),
                     "runs": len(run_counts), "valid_trials_per_run": describe(list(run_counts.values())),
                     "runs_per_subject_session": {"/".join(k): v for k,v in sorted(sessions.items())}},
        "nutrition_and_controls": numeric,
        "kcal_density_times_portion_max_absolute_discrepancy": max(energy_errors),
        "categories": dict(Counter(r["category_name"] for r in images)),
        "candidate_mismatch_pool_audit": {"rule": "Same existing category_name and processing; different kcal/100g after rounding to one decimal. This is only a metadata feasibility count, not validated plausibility or a final permutation.",
                                          "groups": group_rows, "candidate_counts": describe(candidate_counts),
                                          "no_different_value_candidates": zero_candidates,
                                          "unlabelled_category_images": [{k:r[k] for k in ('image_id','food_name','kcal_100g')} for r in images if r['category_name']=='未标注'],
                                          "missing_processing_images": [r['image_id'] for r in images if not r['processing']]},
        "images": {"existing_count": sum(len(v) for v in image_hash_groups.values()),
                   "exact_duplicate_bytes": [v for v in image_hash_groups.values() if len(v)>1],
                   "repeated_normalized_food_names": duplicate_names,
                   "images_in_repeated_name_groups": sum(len(v) for v in duplicate_names.values()),
                   "note": "No perceptual near-duplicate or semantic subgroup review performed"},
        "brain_mapping": {"rows": len(mapping), "distinct_beta_files": len({r['beta_file_local'] for r in mapping}),
                          "distinct_spm_files": len({r['spm_mat_local'] for r in mapping}),
                          "external_files_missing": external_missing,
                          "estimated_successfully": dict(Counter(r['estimated_successfully'] for r in mapping)),
                          "legacy_quality_labels_not_applied": dict(Counter(r['quality_label'] for r in mapping)),
                          "sampled_headers": sum(len(v) for v in geometries.values()),
                          "header_geometries": [{"header": json.loads(k), "runs_count": len(v)} for k,v in geometries.items()],
                          "header_errors": header_errors,
                          "limitation": "One beta header per run inspected; no full-volume ROI quality/reliability audit, no SPM.mat content verification, no beta response extraction"},
        "migration_verification": verification,
        "errors": sorted(set(errors)),
    }
    input_files = list(paths.values()) + [config_path, ROOT / "archive/source_code/run_minimal_lsa.m",
                                         ROOT / "archive/source_code/dataset_author/fig2cdef_EstimateGLMsROI.m"]
    report["provenance"] = {"git_commit_before_audit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                            "python": platform.python_version(), "seed": None,
                            "script_sha256": digest(Path(__file__)),
                            "input_hashes": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in input_files]}
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    (output / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2)+"\n")
    (output / "run_log.txt").write_text("Read-only audit completed. See audit.json for scope, hashes and limitations.\n"+verifier.stderr)
    print(json.dumps({"output": str(output), "errors": report["errors"],
                      "nutrition": {k: numeric[k] for k in ('kcal_100g','fat_100g','fibre_100g')},
                      "categories": report["categories"], "ratings_per_image": report["behavior"]["ratings_per_image"],
                      "runs": len(run_counts), "valid_trials_per_run": report["behavior"]["valid_trials_per_run"],
                      "mismatch_pool_unmatched": zero_candidates, "mismatch_pool_counts": describe(candidate_counts),
                      "beta_headers": report["brain_mapping"]["header_geometries"],
                      "external_missing_count": len(external_missing), "legacy_quality": report["brain_mapping"]["legacy_quality_labels_not_applied"],
                      "duplicate_image_groups": report["images"]["exact_duplicate_bytes"],
                      "duplicate_name_groups": len(duplicate_names)}, ensure_ascii=False, indent=2))
    if errors or external_missing or header_errors or verifier.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
