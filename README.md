# foodLLM

[中文](README.zh-CN.md)

Food representations in vision and vision–language models: predicting individual rating deviations on held-out foods and comparing representations with human fMRI responses.

**Updated: 2026-10-24 · Latest revision analyses completed**

## Study

Qwen3.5-9B, CLIP and DINOv2 are compared using 29 participants, 562 food images and 15,702 ratings, with five-fold food-group-held-out evaluation. The target is each person's rating minus the available other participants' mean for that image. Model selection and preprocessing use training data only.

## Main findings

- With readout selection within training data, individual-deviation correlations are **0.454 for Qwen, 0.435 for CLIP and 0.429 for DINOv2**. Adding Qwen to the joint visual baselines improves MSE by **0.00938**. Fixed R0 readouts yield similar correlations of 0.423–0.426, so the comparison depends on the readout.
- After controlling a rating-scale baseline, Qwen's partial correlation is **0.418**. Its paired differences relative to CLIP and DINOv2 are **0.0197 [0.0060, 0.0344]** and **0.0233 [0.0082, 0.0394]** in conditional 95% bootstrap intervals.
- Within Qwen, adding representations after language-model processing improves held-out brain prediction under the tested controls. The higher-visual conditional gain is **ΔR² = 0.01090**. In a separate common-SST decomposition, **81.44%** of the combined improvement comes from the regional-mean component; this percentage does not decompose the original voxel-averaged ΔR².

These are predictive results for the specified data and readouts. Behavioral and neural gains were measured separately; they do not establish a shared mechanism, language causality or regional superiority.

[Latest results and methods](docs/results.md) · [Main estimates and intervals](results/main_intervals.csv) · [Revision analysis summary](results/revision_summary.csv)

## Scope

This repository presents the latest findings and aggregate tables. Raw data, individual predictions, training pipelines, model weights and manuscript drafts are maintained in the research environment and are not included here. The full behavioral dataset contains 568 images and 15,869 ratings; the common analysis set omits six images with missing fibre values. The tables alone do not reproduce the full experiments. Redistribution terms have not been specified.
