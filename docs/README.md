# 图表查看与复现

[项目首页](../README.zh-CN.md) · [结果说明](results.md)

## 查看或下载原图

原图位于 [v03 figures](../outputs/20260918_控制实验报告_v03/figures)。PNG 适合预览，SVG 适合排版；下载时使用 GitHub 文件页的原始文件下载入口。

| 内容 | 文件 |
| --- | --- |
| A/B 的 41 区逐层差值热图 | `all41_A.png/svg`、`all41_B.png/svg` |
| 四种营养的绝对 RSA 与条件差值 | `absolute_0…3.png/svg`、`contrasts_0…3.png/svg` |
| 41 个区域及 5 组汇总的完整曲线 | `region_<ROI>.png/svg`，见[逐区域索引](../outputs/20260918_控制实验报告_v03/逐区域曲线索引.md) |
| 逐人分布 | `participants_0…3_all32.png/svg`；探索窗口使用 `deep17_32` |
| 跨提示方向、随机区域与测量背景 | `AB_direction_all32.png/svg`、`random_0…3.png/svg`、`quality_all41.png/svg` |

## 从仓库中的 CSV 重绘

在仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
python scripts/reproduce_figures.py --check-only
python scripts/reproduce_figures.py --output outputs/reproduced_figures
```

输出目录须为新目录。脚本生成 48 组 PNG/SVG：

- `region_<ROI>`：41 个区域及 5 组汇总各一页，包含四种营养、A/B 提示、32 层绝对 RSA 与四项差值。
- `all41_A_all32`、`all41_B_all32`：全 32 层平均差值热图；同一对比的色标在 A/B 间一致。
- `reproduction.json`：输入 SHA-256、依赖版本和数值核对误差。

重绘使用原表中的参与者均值，采用新的版式；全层平均热图将原逐层结果汇总为每区一个值。原版逐人图、N/P 实现范围及逐层热图可从上方目录直接下载。

## 数据与核对

| 输入 | 内容 |
| --- | --- |
| [absolute_curves.csv](../outputs/20260918_控制RSA_v01/absolute_curves.csv) | 11,776 行，逐层 V、T、N、P 绝对 RSA 与实现范围 |
| [contrast_curves.csv](../outputs/20260918_控制RSA_v01/contrast_curves.csv) | 49,152 行，逐层四项差值及参与者四分位数，含两组区域间差值 |
| [parallel_region_summary.csv](../outputs/20260918_控制实验报告_v03/parallel_region_summary.csv) | 736 行，全层与探索窗口摘要 |
| [index.json](../outputs/20260918_控制RSA_v01/index.json) | 提示、营养、区域和参与者索引 |

脚本核对全部条件差值能否从绝对分数重建，以及逐层均值能否复现两种窗口摘要。图中 RSA 和差值乘以 1,000，CSV 保留原值。

完整模型推理和脑 RSA 的输入要求见[数据与复现范围](../SHARING.md)。
