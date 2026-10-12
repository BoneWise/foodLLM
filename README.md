# Nutrient-Informed Food Representations: Probing LLM–Brain Alignment

[中文说明](README.zh-CN.md)

This project examines how nutrient information in model inputs changes the alignment between Qwen3.5-9B food representations and human fMRI patterns. Layerwise representational similarity analysis (RSA) compares image-only, matched-nutrient, unrelated-text and mismatched-nutrient conditions.

The current analysis covers **29 participants, 401 food images, 32 model layers and 41 brain regions**, with two prompt wordings and four nutrients: protein, fat, carbohydrate and fibre. OFC, vmPFC, visual and randomly sampled HCP regions provide regional comparisons.

## Results

The descriptive results show nutrient-, region-, layer- and prompt-dependent changes in alignment. Protein and fat produce positive mean changes relative to image-only inputs in several prefrontal regions; carbohydrate and fibre show mixed regional patterns. Matched-versus-mismatched nutrient differences have a distinct distribution from the changes relative to image-only inputs.

[Results and interpretation](docs/results.md) · [All regional curves](outputs/20260918_控制实验报告_v03/逐区域曲线索引.md)

## Reproduce the figures

The repository includes exported RSA tables and the original PNG/SVG figures. Redraw all 46 regional and summary curves, plus two 41-region heatmaps, on a CPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
python scripts/reproduce_figures.py --output outputs/reproduced_figures
```

[Figure guide](docs/README.md) explains the outputs, original figures and numerical checks. Recomputing model features and brain RSA requires the source data and caches listed in [SHARING.md](SHARING.md).

