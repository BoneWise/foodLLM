"""Create a new report from existing v09 exports; never overwrite v01 evidence."""
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/20260918_控制实验报告_v01'
RSA = ROOT / 'outputs/20260918_控制RSA_v01'
OUT = ROOT / 'outputs/20260918_控制实验报告_v02'
NOTE = ROOT / 'docs/research/20260918_区域控制对照口径修订_v09_1.md'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def table(headers, data):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in data])


def rebase(text):
    # Sibling report directories have identical depth; only local v01 assets move.
    def change(m):
        target = m.group(1)
        if not target.startswith(('../', '#', 'http:', 'https:')):
            target = '../' + OLD.name + '/' + target
        return '](' + target + ')'
    return re.sub(r'\]\(([^)]+)\)', change, text)


def main():
    if OUT.exists():
        raise SystemExit('Output already exists; preserve it and create a new version.')
    protected = sorted([p for folder in (OLD, RSA) for p in folder.rglob('*') if p.is_file()])
    protected += [ROOT / 'configs/controls_v09_execution_v01.json',
                  ROOT / 'docs/research/20260918_无关词与随机分区健壮性设计_v09.md',
                  ROOT / 'scripts/write_controls_report_v09.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    index = json.loads((RSA / 'index.json').read_text())
    summary = rows(RSA / 'window_summary.csv')
    lookup = {(r['prompt'], r['nutrient'], r['ROI'], r['window'], r['contrast']): float(r['mean']) for r in summary}
    curves = defaultdict(list)
    for r in rows(RSA / 'absolute_curves.csv'):
        curves[(r['prompt'], r['nutrient'], r['ROI'])].append(r)
    labels = {'protein_100g': '蛋白质', 'fat_100g': '脂肪', 'carbs_100g': '碳水', 'fibre_100g': '纤维'}
    series = index['series']
    assert len(series) == 46 and len(index['parcels']) + 3 == 41
    assert len(index['subject_ids']) == 29 and len(index['image_ids']) == 401
    exported = []
    max_error = 0.0
    for nut in index['nutrients']:
        for pr in index['prompts']:
            for roi in series:
                source = curves[(pr, nut, roi)]
                assert sorted(int(r['layer']) for r in source) == list(range(1, 33))
                for window, layers in index['windows'].items():
                    selected = [r for r in source if int(r['layer']) in layers]
                    absolute = {k: math.fsum(float(r[k]) for r in selected) / len(selected)
                                for k in ['V', 'T', 'N_mean', 'P_mean']}
                    contrasts = {c: lookup[(pr, nut, roi, window, c)]
                                 for c in ['T-V', 'N-V', 'T-N', 'T-meanP']}
                    calculated = {'T-V': absolute['T'] - absolute['V'],
                                  'N-V': absolute['N_mean'] - absolute['V'],
                                  'T-N': absolute['T'] - absolute['N_mean'],
                                  'T-meanP': absolute['T'] - absolute['P_mean']}
                    max_error = max(max_error, *(abs(calculated[c] - contrasts[c]) for c in contrasts))
                    exported.append(dict(prompt=pr, nutrient=nut, ROI=roi, window=window,
                                         **absolute, **contrasts))
    assert max_error < 1e-12
    assert len(exported) == 736
    OUT.mkdir()
    with (OUT / 'parallel_region_summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(exported[0])); w.writeheader(); w.writerows(exported)
    headers = ['区域', '提示', 'V', 'T', '平均N', '平均P', 'T−V', 'N−V', 'T−N', 'T−P']
    fields = ['V', 'T', 'N_mean', 'P_mean', 'T-V', 'N-V', 'T-N', 'T-meanP']
    def numeric_table(nut, window, rois):
        by_key = {(r['ROI'], r['prompt']): r for r in exported
                  if r['nutrient'] == nut and r['window'] == window}
        return table(headers, [[roi, pr] + [f'{1000 * by_key[(roi, pr)][k]:+.3f}' for k in fields]
                               for roi in rois for pr in index['prompts']])
    appendix = ['# 全部区域并列摘要',
                '所有数值均为ρ或ρ差值×1,000；包含41套区域及原5组摘要，不把重叠摘要视作独立样本。'
                '全层与L17–32探索窗口分开。主报告见[结果报告](结果报告.md)。']
    for window in index['windows']:
        for nut, label in labels.items():
            appendix += [f'## {label} · {window}', numeric_table(nut, window, series)]
    (OUT / '全部区域并列摘要.md').write_text('\n\n'.join(appendix) + '\n')
    old = (OLD / '结果报告.md').read_text()
    def section(start, end=None):
        text = old.split(start, 1)[1]
        if end:
            text = text.split(end, 1)[0]
        return rebase(text.strip())
    report = ['''# 输入条件与区域控制对照：v09 修订报告 v02

日期：2026-09-18。**OFC、vmPFC、视觉区域均作为区域控制对照，相互参照；随机HCP区域扩展比较范围。不预设某一区域必须更强。**

本版按用户明确纠正并批准的口径重写解释与阅读顺序。原报告曾错误加入OFC特异性的方向性强假设，现予纠正。既有实验、冻结配置和数值不变；这不是新增实验或追认预注册假设。详见[口径修订记录](../../docs/research/20260918_区域控制对照口径修订_v09_1.md)。

## 研究问题与控制作用

加入营养信息后，模型表征与不同脑区表征的对齐如何变化？这些变化与无关文本、错配营养信息条件相比有什么差异，在OFC、vmPFC、视觉及随机参照区域之间呈现怎样的分布？

输入控制包括图片基线V、正确营养T、五套错配营养P和三套本次无关文本N。区域控制比较不同区域的分数和条件差值；任何区域都不预设应有或应无效应。视觉或随机区域出现较大变化，同样是区域比较的结果。

## 结果概览

- 条件差值随营养、区域、层与提示变化，应结合各区域的绝对分数解释，不能用一个区域概括整个项目。
- 在OFC九区汇总中，蛋白质、脂肪的全层T−V及T−N在A/B均为正；碳水的T−V均负而T−N为正；纤维T−N随提示变号。
- 高阶视觉中，脂肪全层T−V及T−N在A/B均为正，数值大于OFC九区对应差值。早中视觉、高阶视觉的分数分开呈现，不能合成一个“视觉效应”。
- 原vmPFC球中的N−V在全层A约−0.00018、B约+0.00108，显示本次无关文本的变化也有区域与提示依赖。其定义与HCP分区汇总有别，结果单列。
- 随机区域提供其他皮层分区的条件变化分布；全部20对结果与原区域一同保留。既有随机九区位置只作补充描述。
- T−P独立于T−N解读。以OFC九区为例，蛋白质／脂肪全层T−P均值约+0.00006至+0.00020，五套P实现范围跨零；其他区域的T−P在下表及全部区域表中并列报告，不由这一例子推断全项目机制。

以上均为描述性结果，没有显著性或等效性检验，也不按某一区域胜出与否判定实验成功或失败。

## 完成范围
''', section('## 一、完成范围', '## 二、主要结果如何读'), '''## 各区域的绝对分数与条件差值

下表所有分数和差值均乘以1,000；`1.000`对应ρ或ρ差值`0.001`，不是百分比或标准化效应量。绝对分数按原逐层群体均值等权汇总；差值直接取原全层窗口表，两者数值一致性另行核验。

T−V为正确营养相对图片基线的变化，N−V为本次无关文本自身的变化，T−N为两类文本之差，T−P为正确与错配营养之差。正差值不自动意味着绝对RSA为正；T−N为正也不自动意味着T−V为正。同一配对下T−V=(N−V)+(T−N)只是分数恒等式，不是因果中介分解。

为便于阅读，主表并列列出既有OFC汇总、原三ROI、10v/s32和两组视觉汇总；选择仅改变展示，不新增ROI或统计主效应。OFC汇总相互重叠，不是独立复制；旧三ROI与HCP分区保持不同测量定义。全部41区域及5组摘要、两个层窗口见[全部区域并列摘要](全部区域并列摘要.md)和[数值表](parallel_region_summary.csv)。''']
    main_rois = ['OFC_nine', 'lateral_OFC', 'medial_OFC', 'lOFC', 'mOFC', 'vmPFC',
                 '10v', 's32', 'early_intermediate_visual', 'higher_visual']
    for nut, label in labels.items():
        report += [f'### {label} · 全32层', numeric_table(nut, 'all32', main_rois)]
    report += ['## 全区域与逐层曲线', '''全部41区域的既有图首先并列提供：

![全部41区域 A](../20260918_控制实验报告_v01/figures/all41_heatmap_A.png)

![全部41区域 B](../20260918_控制实验报告_v01/figures/all41_heatmap_B.png)

下列曲线保留既有绘图顺序，该顺序不定义主要区域。''',
               section('### 完整层曲线', '## 三、无关文本自身影响了哪些区域？'),
               '## 无关文本自身的区域差异（N−V）',
               section('## 三、无关文本自身影响了哪些区域？', '## 四、OFC 的变化是否相对高阶视觉更强？'),
               '## 实现范围与个体分布', section('## 六、实现范围和个体分布', '## 七、控制条件及其限制'),
               '## 控制条件及解释边界', section('## 七、控制条件及其限制', '## 八、数据完整性与复现'),
               '## 补充描述：既有区域差值与随机分区参照',
               '以下保留原来计算的OFC−高阶视觉差值及九区参照位置，仅用于描述区域分布。差值的减法顺序不代表预期方向；本次不新增其他成对比较或显著性检验。',
               '### OFC九区−高阶视觉的既有差值',
               section('## 四、OFC 的变化是否相对高阶视觉更强？', '## 五、20对随机分区提供了什么参照？').replace(
                   '区域差值为正也可能来自视觉下降，须结合OFC自身T−V和绝对RSA。不能由一个区域为正、另一区域为负直接推断显著区域特异性。',
                   '差值为正表示前者数值大于后者，也可能来自后者下降更多；正负方向均须结合两侧自身变化和绝对RSA。这里未检验区域差异的显著性。'),
               '### 20对随机区域及既有匹配描述',
               section('## 五、20对随机分区提供了什么参照？', '## 六、实现范围和个体分布').replace(
                   '### OFC各分区的可比参照数量', '#### OFC分区及10v/s32的既有可比参照数量').replace(
                   '### OFC九区摘要与九区参照摘要', '#### 既有九区汇总的描述性位置').replace(
                   '旧3个ROI只作标注连续性参照，不给严格同单位排名',
                   '旧3个ROI参与区域控制对照；因测量定义不同，在这一匹配排名中仅标注、不作严格同单位排名'),
               '## 数值来源与版本验证',
               '本次仅重排已有数值并修订解释。旧实验核查记录描述的是原运行；本次文件保全、链接、全区域覆盖与摘要一致性见[修订核查](revision_audit.json)。原v01报告及原报告生成脚本保留为历史，不再作为当前口径入口。',
               section('## 八、数据完整性与复现')]
    (OUT / '结果报告.md').write_text('\n\n'.join(report) + '\n')
    shutil.copy2(__file__, OUT / 'script_snapshot.py')
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert before == after
    broken = []
    for path in [OUT / '结果报告.md', OUT / '全部区域并列摘要.md', NOTE]:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('http:', 'https:', '#')) or target == 'revision_audit.json':
                continue
            if not (path.parent / target.split('#')[0]).exists():
                broken.append([str(path.relative_to(ROOT)), target])
    assert not broken, broken
    audit = dict(status='complete', kind='user_approved_interpretation_revision_no_new_experiment',
                 created=datetime.now().astimezone().isoformat(),
                 git_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 git_status_note='Pre-existing uncommitted work retained; exact revision script snapshotted; no remote push.',
                 dependencies={'python': platform.python_version(), 'libraries': 'standard_library_only'},
                 parameters={'windows': index['windows'], 'main_table_ROIs': main_rois,
                             'all_series': series, 'display_scale': 1000, 'rounding_decimals': 3},
                 random_seed=None, random_seed_reason='No new random process; experiment seeds remain in frozen config.',
                 inference_run=False, RSA_recomputed=False,
                 protected_files_sha256=before, protected_files_unchanged=True,
                 input_sha256={str(NOTE.relative_to(ROOT)): sha(NOTE)},
                 summary_rows=len(exported), actual_regions=41, existing_group_summaries=5,
                 absolute_to_original_contrast_max_error=max_error,
                 all_local_links_exist=True,
                 output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()})
    (OUT / 'revision_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ['status', 'summary_rows', 'protected_files_unchanged',
                                          'absolute_to_original_contrast_max_error', 'all_local_links_exist']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
