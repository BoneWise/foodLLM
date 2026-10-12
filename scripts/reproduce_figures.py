"""Redraw layerwise RSA figures from the CSV tables included in this repository."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/20260918_控制RSA_v01"
SUMMARY = ROOT / "outputs/20260918_控制实验报告_v03/parallel_region_summary.csv"
NAMES = ["Protein", "Fat", "Carbohydrate", "Fibre"]
CONTRASTS = ["T-V", "N-V", "T-N", "T-meanP"]
FIELDS = ["V", "T", "N_mean", "P_mean"]
LABELS = ["Image (V)", "Matched nutrient (T)", "Unrelated text (N)", "Mismatched nutrient (P)"]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_and_check():
    index = json.loads((SOURCE / "index.json").read_text())
    prompts, nutrients, regions = (index[k] for k in ("prompts", "nutrients", "series"))
    assert len(prompts) == 2 and len(nutrients) == 4 and len(regions) == 46
    absolute, contrasts = {}, {}
    for row in read_csv(SOURCE / "absolute_curves.csv"):
        key = (row["prompt"], row["nutrient"], row["ROI"], int(row["layer"]))
        assert key not in absolute, f"Duplicate absolute score: {key}"
        absolute[key] = row
    for row in read_csv(SOURCE / "contrast_curves.csv"):
        key = (row["prompt"], row["nutrient"], row["ROI"], int(row["layer"]), row["contrast"])
        assert key not in contrasts, f"Duplicate contrast: {key}"
        contrasts[key] = float(row["mean"])
    assert len(absolute) == 2 * 4 * 46 * 32
    assert len(contrasts) == 2 * 4 * 48 * 32 * 4
    error = 0.0
    for prompt in prompts:
        for nutrient in nutrients:
            for region in regions:
                for layer in range(1, 33):
                    key = (prompt, nutrient, region, layer)
                    row = absolute[key]
                    v, t, n, p = (float(row[f]) for f in FIELDS)
                    assert np.isfinite([v, t, n, p]).all()
                    for contrast, expected in zip(CONTRASTS, [t-v, n-v, t-n, t-p]):
                        error = max(error, abs(expected - contrasts[key + (contrast,)]))
    for prompt in prompts:
        for nutrient in nutrients:
            for visual in ["higher_visual", "early_intermediate_visual"]:
                for layer in range(1, 33):
                    for contrast in CONTRASTS:
                        ofc = contrasts[(prompt, nutrient, "OFC_nine", layer, contrast)]
                        reference = contrasts[(prompt, nutrient, visual, layer, contrast)]
                        exported = contrasts[(prompt, nutrient, "OFC_minus_"+visual, layer, contrast)]
                        error = max(error, abs(ofc-reference-exported))
    summaries = read_csv(SUMMARY)
    assert len(summaries) == 2 * 4 * 46 * 2
    for row in summaries:
        layers = range(1, 33) if row["window"] == "all32" else range(17, 33)
        keys = [(row["prompt"], row["nutrient"], row["ROI"], layer) for layer in layers]
        for field in FIELDS:
            error = max(error, abs(np.mean([float(absolute[k][field]) for k in keys]) - float(row[field])))
        for contrast in CONTRASTS:
            error = max(error, abs(np.mean([contrasts[k + (contrast,)] for k in keys]) - float(row[contrast])))
    assert error < 1e-12, f"Exported tables disagree: {error}"
    return index, absolute, contrasts, error


def save(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduced_figures")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    index, absolute, contrasts, error = load_and_check()
    print(f"Verified 11,776 absolute-score rows, 49,152 contrast rows and 736 window summaries; max error {error:.3g}.")
    if args.check_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    layers = np.arange(1, 33)
    for region in index["series"]:
        fig, axes = plt.subplots(4, 4, figsize=(16, 12), layout="constrained", sharex=True)
        for ni, nutrient in enumerate(index["nutrients"]):
            for pi, prompt in enumerate(index["prompts"]):
                keys = [(prompt, nutrient, region, int(layer)) for layer in layers]
                ax = axes[ni, 2*pi]
                for field, label in zip(FIELDS, LABELS):
                    ax.plot(layers, [float(absolute[k][field])*1000 for k in keys], label=label)
                ax.set_title(f"{NAMES[ni]} | Prompt {prompt}")
                ax.set_ylabel("RSA (Spearman rho × 1,000)")
                ax = axes[ni, 2*pi+1]
                for contrast in CONTRASTS:
                    ax.plot(layers, [contrasts[k+(contrast,)]*1000 for k in keys], label=contrast)
                ax.set_title(f"{NAMES[ni]} | Prompt {prompt}: differences")
                ax.set_ylabel("RSA difference × 1,000")
        for ax in axes.flat:
            ax.axhline(0, color="0.65", lw=0.6)
            ax.set_xticks([1, 8, 16, 24, 32])
            ax.grid(alpha=0.15)
        for ax in axes[-1]:
            ax.set_xlabel("Model layer")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].legend(fontsize=7)
        fig.suptitle(f"Nutrient-Informed Food Representations: LLM–Brain Alignment\n{region} | Participant means (n = 29)", fontsize=15)
        save(fig, args.output, f"region_{region}")
    # The first 41 series are individual regions; the last five are regional summaries.
    regions = index["series"][:41]
    for prompt in index["prompts"]:
        fig, axes = plt.subplots(1, 4, figsize=(15, 13), layout="constrained", sharey=True)
        for ax, contrast in zip(axes, CONTRASTS):
            def matrix(p):
                return np.array([[np.mean([contrasts[(p, n, r, int(layer), contrast)] for layer in layers])*1000
                                  for n in index["nutrients"]] for r in regions])
            limit = max(float(np.abs(matrix(p)).max()) for p in index["prompts"])
            im = ax.imshow(matrix(prompt), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
            ax.set_xticks(range(4), NAMES, rotation=45, ha="right")
            ax.set_yticks(range(41), regions)
            ax.set_title(contrast)
            fig.colorbar(im, ax=ax, shrink=0.6, label="Mean RSA difference × 1,000")
        fig.suptitle(f"Nutrient-Informed Food Representations: LLM–Brain Alignment\nPrompt {prompt} | All 32 layers | 41 regions", fontsize=15)
        save(fig, args.output, f"all41_{prompt}_all32")
    inputs = [SOURCE / "index.json", SOURCE / "absolute_curves.csv", SOURCE / "contrast_curves.csv", SUMMARY]
    manifest = {"source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "max_table_error": error, "figure_pairs": 48,
                "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "description": "Redrawn participant-mean curves and all-layer heatmaps from exported RSA tables."}
    (args.output / "reproduction.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Saved 48 PNG/SVG pairs to {args.output}")


if __name__ == "__main__":
    main()
