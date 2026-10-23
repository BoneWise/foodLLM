# foodLLM

[中文说明](README.zh-CN.md)

This project studies food representations in vision and vision–language models, the prediction of individual food ratings, and their relationship to human fMRI responses.

## Latest progress — 2026-10-24

The newer analyses compare Qwen3.5-9B, CLIP and DINOv2 on **29 participants, 562 food images and 15,702 ratings**, with food-group-held-out evaluation. The full behavioral dataset contains 568 images and 15,869 ratings; the common analysis set excludes six images with missing fibre values.

With readout selection performed within training data, Qwen predicts individual rating deviations at **r = 0.454**, compared with **0.435 for CLIP** and **0.429 for DINOv2**. Adding Qwen to the joint visual baselines reduces MSE by **0.00938**. The advantage depends on the readout: fixed R0 results are approximately 0.423–0.426 across models.

Within Qwen, representations after language-model processing also add held-out brain-prediction information under the tested controls. These behavioral and neural gains were measured separately; they do not establish a shared mechanism, language causality or a privileged brain region. The current public update includes aggregate results and conditional bootstrap intervals; the newer training pipelines and prediction arrays remain in the research environment.

[October progress and interpretation / 最新进展](docs/progress-2026-10.md) · [26 aggregate estimates and intervals](outputs/20261023_summary/intervals.csv) · [Sharing and reproducibility scope](SHARING.md)

## Earlier nutrient-conditioned RSA — September 2026

This earlier phase examined how nutrient information in model inputs changes the alignment between Qwen3.5-9B food representations and human fMRI patterns. Layerwise representational similarity analysis (RSA) compares image-only, matched-nutrient, unrelated-text and mismatched-nutrient conditions.

The earlier RSA analysis covers **29 participants, 401 food images, 32 model layers and 41 brain regions**, with two prompt wordings and four nutrients: protein, fat, carbohydrate and fibre. OFC, vmPFC, visual and randomly sampled HCP regions provide regional comparisons.

### RSA results

The descriptive results show nutrient-, region-, layer- and prompt-dependent changes in alignment. Protein and fat produce positive mean changes relative to image-only inputs in several prefrontal regions; carbohydrate and fibre show mixed regional patterns. Matched-versus-mismatched nutrient differences have a distinct distribution from the changes relative to image-only inputs.

[Results and interpretation](docs/results.md) · [All regional curves](outputs/20260918_控制实验报告_v03/逐区域曲线索引.md)

### Reproduce the earlier RSA figures

The repository includes exported RSA tables and the original PNG/SVG figures. Redraw all 46 regional and summary curves, plus two 41-region heatmaps, on a CPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
python scripts/reproduce_figures.py --output outputs/reproduced_figures
```

[Figure guide](docs/README.md) explains the outputs, original figures and numerical checks. Recomputing model features and brain RSA requires the source data and caches listed in [SHARING.md](SHARING.md).

