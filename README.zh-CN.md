# 营养信息条件下的食物表征：大语言模型与脑表征对齐

[English](README.md)

本项目考察：向模型提供营养信息后，其食物表征与人类 fMRI 表征的对应关系如何变化？研究使用 Qwen3.5-9B 的逐层隐藏表征，通过表征相似性分析（RSA）比较图片基线、匹配营养、无关文本和错配营养四类输入。

当前分析覆盖 **29 名参与者、401 张食物图片、32 个模型层和 41 个脑区**，包含两种提示措辞，以及蛋白质、脂肪、碳水化合物和纤维四种营养属性。OFC、vmPFC、视觉区域与随机抽取的 HCP 分区共同用于比较变化的区域分布。

## 结果

描述性结果显示，对齐变化随营养、脑区、层和提示而变化。蛋白质与脂肪在多个前额叶区域呈现相对图片基线的正向均值变化；碳水化合物与纤维呈现不同的区域模式。正确匹配与错配营养之间的差异，需要结合图片基线和无关文本条件共同解释。

[结果说明](docs/results.md) · [全部区域曲线](outputs/20260918_控制实验报告_v03/逐区域曲线索引.md)

## 图表复现

仓库包含 RSA 导出表和原始 PNG/SVG 图。以下命令使用 CPU 重绘 46 套区域及汇总曲线，以及两张 41 区热图：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
python scripts/reproduce_figures.py --output outputs/reproduced_figures
```

[图表指南](docs/README.md)说明原图位置、输出文件和数值核对方式。重新提取模型特征并计算脑 RSA 所需的输入见[数据与复现范围](SHARING.md)。
