对，你的理解基本正确。你原来的 v06 **主分析本身就是直接使用单试次 trial β 的多体素 pattern 构建 RDM**，并没有先对 β 做一堆额外变换。

你原方案已经包含的是“入口 QC”，不是“改变数据的 preprocessing”：

* 检查 ROI 覆盖、固定 voxel identity/order；
* 检查 β 文件与 trial/image ID 对应；
* 检查 finite values、常数 pattern、几何一致性；
* 计算 session RDM 作为描述性 QC；
* 主分析仍然直接用原始单试次 β pattern：

  $$
  D^{brain}(i,j)=1-r(\mathbf b_i,\mathbf b_j)
  $$

没有 PCA、白化、session centering、voxel-wise normalization 等。

所以我前面第 3 点实际上不是让你**增加**一步，而是在说：

> **不要增加这些额外步骤，先保持 raw trial-beta RSA。**

这一点你原方案已经做对了。

---

## 我现在反而建议进一步把 v06 简化

正式第一轮 RSA 可以只分成三层。

### 第一层：确认输入能用

这是正式 RSA 前真正必须做的。

只检查：

1. single-trial β 是否确实来自 **unsmoothed** 数据；
2. 每个参与者 vmPFC ROI 的 voxel 数是否合理；
3. 同一参与者所有 trial 的 voxel identity/order 是否完全一致；
4. β 是否 finite；
5. 是否存在整个 vmPFC pattern 方差为 0、无法计算 correlation 的 trial；
6. image ID / trial / β 对应是否正确。

这些都是**数据有效性检查**。

不涉及科学结果，也不改变数据。

我认为这一层做完就可以开始 RSA。

---

# 第二层：直接跑最原始、最容易解释的 RSA

不要先 centering，不要 whiten，不要 partial RSA。

直接：

$$
D^{brain}_s(i,j)
=
1-r(\mathbf b_{s,i},\mathbf b_{s,j})
$$

模型：

$$
D^{model}_{l,c}(i,j)
=
1-r(\mathbf h_{l,c,i},\mathbf h_{l,c,j})
$$

然后：

$$
A_{s,l,c}
=
\rho_{Spearman}
(D^{brain}_s,D^{model}_{l,c})
$$

只画：

$$
V,\quad T,\quad P
$$

以及：

$$
T-V
$$

$$
T-P.
$$

这就是第一轮最重要的结果。

我甚至建议第一轮先只跑 **Prompt A**。

因为 A 是 primary。

等确定整个 RSA pipeline 没问题，再把 B 跑出来做 robustness。虽然计算很便宜，但分析认知上会简单很多。

---

# 第三层：QC 也先只“看”，不要立即修正

你原来的 session QC 可以保留。

我前面建议增加的 run QC，我现在会进一步收缩成：

> **只计算和报告，不自动触发任何 preprocessing。**

例如建立：

$$
D^{run}(i,j)=
\begin{cases}
0,&same\ run\\
1,&different\ run
\end{cases}
$$

然后看看：

$$
RSA(D^{brain},D^{run})
$$

有多大。

session 同理。

目的只是回答：

> brain RDM 里面有没有非常明显的扫描结构？

第一轮不要因为有一点 run/session structure 就立刻：

* session-center；
* remove run mean；
* whiten；
* 删除 same-run pairs。

这些都会改变 RDM 本身。

如果 primary RSA 有结果，同时 run/session QC 显示很弱，那么事情非常简单。

如果 QC 显示非常强，再决定 sensitivity analysis。

---

## 所以我会把你原来的“两种 session sensitivity 必须都预先报告”降级

v06 目前写：

1. session-level pattern centering；
2. within-session-only RSA；

都预先运行。

我现在觉得对于**第一轮探索这个 RSA 方法是否适合你们数据**而言，有点过重。

特别是 session centering：

$$
b_{trial}
-
\overline b_{session}
$$

这并不是一个纯粹 QC。

它已经在**改变 neural representation**。

而且它有可能同时删除真正存在的跨 session food geometry。

所以我会改成：

> **Primary：raw trial β big-RDM。**
> **QC：session/run identity 与 brain RDM 的关系。**
> **只有 QC 表明扫描结构明显时，再运行预先定义的 sensitivity analysis。**

这样最符合你说的：

> 刚开始不要增加太多步骤。

---

# 关于统计：我同意你，现在考虑得有点太远

尤其这一大段：

* 50,000 sign flips；
* 64-dimensional max statistic；
* studentized statistic；
* 10,000 bootstrap；
* simultaneous confidence band；
* coverage simulation……

我认为**目前完全没必要先实现**。

不是这些方法有问题，而是你现在连：

> raw vmPFC RSA 到底有没有稳定结构

都还不知道。

合理顺序应该是：

### Phase 1：measurement / pipeline validation

先得到每个人每层：

$$
A^V,A^T,A^P
$$

画出来。

确认：

* RDM 算法正确；
* participant scores 不是 NaN；
* 32 层曲线不是异常；
* A/B 基本能运行；
* 五套 P 都有结果；
* session/run QC 没有明显灾难。

### Phase 2：描述性结果

看：

$$
\overline{T-V}
$$

$$
\overline{T-P}
$$

以及 individual participant distribution。

先回答：

> effect 量级到底是 \(10^{-4}\)、\(10^{-2}\)，还是接近 0？

### Phase 3：再冻结正式 inferential procedure

这个时候再决定：

* participant sign-flip；
* layer correction；
* confidence interval；
* 是否需要 stimulus generalization。

更合理。

---

## 但有两个统计原则现在就应该固定

虽然不用马上写 50,000 次 permutation，但以下两件事情现在就确定，避免以后因为结果选择统计单位：

### 1. 推断单位是 participant

不是：

* 121,278 个 RDM pairs；
* 32 layers；
* 五套 P。

真正独立的人类单位仍然是：

$$
n=29.
$$

### 2. 主 contrast 现在固定

$$
T-V
$$

和：

$$
T-\overline P.
$$

这一点不要等看 RSA 曲线以后改。

至于之后到底 maxT、cluster across layer 还是同时带，可以后面再决定并诚实记录是在结果可见后确定的。

由于这本身已经是 post-MSE follow-up，本来也不需要把它伪装成完全盲态 confirmatory analysis。

---

# 我还会删除一个目前没必要的复杂内容：behavior bootstrap

现在 wanting RDM 是：

$$
D^{wanting}_{ij}
=
|H_i-H_j|.
$$

第一轮你完全可以把它当：

> **一个固定的 behavioral reference RDM**

然后每层计算：

$$
RSA(ModelRDM,WantingRDM).
$$

先画：

$$
V/T/P
$$

即可。

现在根本不用立刻做：

> resample 29 participants → 重算 493 个均值 → 重算 RDM → 10,000 bootstrap。

因为 wanting 在当前 RSA 项目里只是一个**支持性的 behavioral anchor**，不是 primary inferential outcome。

第一轮甚至只需要展示它。

后面论文真正需要行为 uncertainty 时再做 bootstrap。

---

# kcal RDM 也一样

继续保留：

$$
D^{kcal}_{ij}
=
|kcal_i-kcal_j|.
$$

但把它当一条水平 reference：

> raw energy geometry 本身与 vmPFC / wanting 有多少 alignment？

不用给它再建一个显著性家族。

它只是帮你理解：

> T alignment 如果上升，是不是可能主要因为模型更明确编码了 kcal dimension。

---

# 我会把 v06 改成非常简洁的 v06.1

## Primary data

**Brain**

raw unsmoothed single-trial LSA β
→ fixed vmPFC ROI
→ correlation-distance big RDM.

**Model**

Qwen3.5-9B
→ Prompt A
→ V / T / P1–P5
→ 32 layers
→ same pre-answer token
→ correlation-distance RDM.

---

## Primary result

对于 participant \(s\)、layer \(l\)：

$$
RSA_{s,l,c}
=
Spearman(RDM^{model}_{l,c},RDM^{brain}_s)
$$

然后报告完整：

$$
V_l,\quad T_l,\quad \bar P_l
$$

和：

$$
\boxed{\Delta_l^{T-V}}
$$

$$
\boxed{\Delta_l^{T-P}}.
$$

这就是主结果。

---

## Behavioral reference

$$
RDM^{wanting}
$$

做完全一样的 V/T/P layer curves。

不做复杂 inferential bootstrap，第一轮作为 reference。

---

## Nutrient reference

$$
RDM^{kcal}.
$$

同样只作为解释性 reference。

---

## QC

正式 RSA 前：

* unsmoothed β；
* fixed voxel set；
* finite；
* constant-pattern；
* trial/image mapping。

RSA 跑完后描述：

* session RDM；
* run RDM。

**不自动改变 primary brain RDM。**

---

## Robustness 暂时只有一个

Prompt B。

如果 A pipeline 运行正常，再原样计算 B。

session-centering、within-session-only、exclude-same-run 全部先写成：

> **reserved sensitivity analyses if acquisition structure appears material。**

不在第一轮全部跑。

---

# 还有一个地方我建议保留你现在的做法：五套 P 先算 RSA 再平均

这个是对的：

$$
A^{P_1},...,A^{P_5}
$$

然后：

$$
\bar A^P
=
\frac{1}{5}\sum_kA^{P_k}.
$$

不要先平均五套 P 的 hidden state/RDM。

因为五套 P 是五个不同的 mismatch realizations。

这是一个值得保留、但不会增加多少复杂度的设计。

---

## 所以最终执行顺序我会改成

1. **核对 unsmoothed β 和 ROI pattern 数据。**
2. **提取 raw trial-level vmPFC patterns。**
3. **构建 29 个完整 big brain RDM。**
4. **构建 Prompt A 的 V/T/P × 32 layer model RDM。**
5. **直接计算 RSA，画 V/T/P、T−V、T−P 曲线。**
6. **计算 wanting/kcal reference curves。**
7. **计算 session/run QC。**
8. **确认一切合理后，跑 Prompt B。**
9. **最后再讨论正式统计推断和是否需要 sensitivity analysis。**

这个顺序我认为最适合你现在的阶段。

你现在最需要知道的不是：

> “怎样做一个统计上无懈可击的最终发表版本？”

而是：

> **“raw vmPFC multivoxel geometry 与 Qwen layer geometry 到底有没有可测、可解释的关系，以及正确 energy information 有没有改变这种关系？”**

先把这个最核心的 measurement 跑出来，再决定后面哪些控制真正值得付出复杂度。
