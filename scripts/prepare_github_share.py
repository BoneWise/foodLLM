"""Build an audited sharing snapshot; never publish or modify source evidence."""
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'outputs/20260918_GitHub分享准备_v01/repository'
TEXT = {'.py', '.md', '.json', '.csv', '.txt', '.sh', '.yaml', '.yml'}
REPORTS = [
    '20260918_控制实验报告_v03', '20260918_控制实验报告_v02',
    '20260918_控制实验报告_v01', '20260918_控制RSA_v01',
    '20260917_HCP_RSA_v01', '20260917_四营养RSA报告_v01',
    '20260917_OFC_kcal报告_v02', '20260917_RSA描述性报告_v01',
]
EXCLUDE = {'docs/服务器连接.md', 'docs/research/20260916_准备完成与Qwen部署记录.md'}
LINK = re.compile(r'(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)')

def sha(b):
    return hashlib.sha256(b).hexdigest()

def sanitize(s):
    # Deployment identifiers only; scientific parameters and numerical results stay intact.
    s = s.replace('/path/to/foodLLM', '/path/to/foodLLM')
    s = s.replace('/path/to/foodLLM', '/path/to/foodLLM')
    s = s.replace('/path/to/external_data', '/path/to/external_data')
    s = s.replace('/path/to/local_data', '/path/to/local_data')
    s = s.replace('YOUR_SERVER_HOST', 'YOUR_SERVER_HOST')
    s = s.replace('ssh YOUR_SERVER_ALIAS', 'ssh YOUR_SERVER_ALIAS')
    return s

def main():
    DEST.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / n for n in ('README.md', 'CHANGELOG.md', 'AGENTS.md')]
    for folder in ('scripts', 'configs', 'docs'):
        sources.extend(p for p in (ROOT / folder).rglob('*')
                       if p.is_file() and p.suffix in TEXT and '__pycache__' not in p.parts
                       and 'migration' not in p.parts and str(p.relative_to(ROOT)) not in EXCLUDE)
    for folder in REPORTS:
        sources.extend(p for p in (ROOT / 'outputs' / folder).rglob('*')
                       if p.is_file() and p.suffix in TEXT | {'.png', '.svg', '.jpg'})
    selected = {p.relative_to(ROOT).as_posix(): p for p in sources}
    missing = []
    source_manifest = []
    for rel, p in sorted(selected.items()):
        original = p.read_bytes()
        b = original
        transformations = []
        if p.suffix in TEXT:
            s = original.decode('utf-8')
            cleaned = sanitize(s)
            if cleaned != s:
                transformations.append('deployment_path_or_host_placeholder')
            s = cleaned
            if p.suffix == '.md':
                def fix_link(m):
                    label, target = m.groups()
                    target = target.strip('<>')
                    if re.match(r'[a-zA-Z]+:', target) or target.startswith('#'):
                        return m.group(0)
                    part = urllib.parse.unquote(target.split('#')[0])
                    resolved = (p.parent / part).resolve()
                    try:
                        destrel = resolved.relative_to(ROOT).as_posix()
                    except ValueError:
                        destrel = 'outside_project'
                    if destrel in selected or (resolved.is_dir() and any(x.startswith(destrel + '/') for x in selected)):
                        return m.group(0)
                    missing.append({'source': rel, 'label': label, 'original_target': sanitize(target)})
                    guide = os.path.relpath(DEST / 'SHARING.md', (DEST / rel).parent)
                    return f'[{label}（本分享包未收录）]({guide}#未收录材料)'
                rewritten = LINK.sub(fix_link, s)
                if rewritten != s:
                    transformations.append('omitted_dependency_links_annotated')
                s = rewritten
            b = s.encode('utf-8')
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b)
        source_manifest.append(dict(path=rel, source_sha256=sha(original), exported_sha256=sha(b),
                                    bytes=len(b), transformations=transformations))
    # The landing page clearly distinguishes the share from the complete local project.
    readme = DEST / 'README.md'
    banner = ('> **私有代码与结果分享快照（2026-09-18）**：最新入口为下方 v03 报告。'
              '此包包含代码、方案、图表及匿名逐人 RSA 派生结果；完整原始数据、缓存与本地 Git 历史未收录。'
              '文件范围、路径替换和复现边界见 [分享说明](SHARING.md)。\n\n')
    readme.write_text(banner + readme.read_text())
    for row in source_manifest:
        if row['path'] == 'README.md':
            row['transformations'].append('share_scope_banner')
            row['exported_sha256'] = sha(readme.read_bytes())
            row['bytes'] = readme.stat().st_size
    (DEST / '.gitignore').write_text('.DS_Store\n__pycache__/\n*.py[cod]\n.venv/\n.env\n.env.*\narchive/\ndata/\nmodels/\n.cache/\n*.npy\n*.npz\n*.nii\n*.nii.gz\n*.safetensors\n')
    packages = ['numpy','pandas','scipy','matplotlib','pillow','nibabel','scikit-learn','openpyxl']
    (DEST / 'requirements-local-analysis.lock.txt').write_text(
        '# Selected analysis packages observed in the local Python 3.12 environment.\n' +
        '\n'.join(f'{n}=={importlib.metadata.version(n)}' for n in packages) + '\n')
    (DEST / 'SHARING.md').write_text('''# 分享范围与复现边界

目标：由用户管理的 huzhe3566 组织下的 foodLLM 私有仓库；指定协作者只读。
本文件记录预定目标，不表示远端创建、上传或邀请已经完成。

## 阅读顺序

1. [当前完整报告](outputs/20260918_控制实验报告_v03/结果报告.md)
2. [获批区域控制对照口径](docs/research/20260918_区域控制对照口径修订_v09_1.md)
3. [完整曲线索引](outputs/20260918_控制实验报告_v03/逐区域曲线索引.md)
4. [当前研究状态](docs/研究状态与下一阶段.md)

OFC、vmPFC、视觉区域均为相互参照的区域控制对照，随机区域扩展参照范围。
无预设区域优越性，不以 OFC 特异性判定成败。当前结果为描述性，正式推断与敏感性未执行。

## 内容

包含当前 scripts、configs、研究文档、v03 的全部 71 组 PNG/SVG、v01/v02 溯源报告、
v09 的绝对分数/条件差值/窗口/匿名逐人 RSA 派生表，以及首页引用的 v06.1/v07/v08 阶段报告。
匿名逐人派生表保留 sub 编号以便配对核查；它们仍属于本次私有分享范围。

## 未收录材料

不含 archive、原图、原始或清理行为评分、事件、脑影像和多体素数据、模型权重、
隐藏状态、RDM/NPY/NPZ 数组、服务器连接记录、部署登录文档及原本地 Git 历史。
原文中的未收录相对链接已加注并指向本节；完整列表见 excluded_links.json。
代码与配置仍可能引用这些本地输入，不能因文件存在便认为全流程可在本包直接运行。

## 完整性与路径替换

source_manifest.json 逐项记录原文件与分享副本的 SHA-256、大小及改动种类；
SHA256SUMS.json 校验最终分享包文件（不含其自身）。原本地证据没有被覆盖。
分享副本将机器路径改为 /path/to/...、服务器地址改为 YOUR_SERVER_HOST。
配置中的占位符需按实际部署配置；历史脚本仍需自行检查入口与输入，不能直接无条件运行。
历史审计中的哈希仍指向原始本地运行证据，不重新伪造为分享副本的哈希。
因此，历史校验脚本可能因未收录文件或副本路径替换而停止；这不等于原实验校验失败。

## 环境与复现

requirements-local-analysis.lock.txt 记录本地 Python 3.12 的主要分析包版本；
configs/qwen35_server_requirements.lock.txt 保留原模型服务器环境记录。
两者为已观察环境，不是一次全新机器安装验证。查看 Markdown、PNG、CSV 无需模型环境。
分享包可以复核导出分数和阅读报告；全量重跑模型/脑 RSA 仍需未收录的授权数据、权重和缓存。
报告生成脚本通常拒绝覆盖既有目录，复跑时须显式准备新输出目录。

本次只制作代码与结果快照，未新增实验、未重算 RSA，也未授予开源许可。
''')
    for name, value in [('source_manifest.json', source_manifest), ('excluded_links.json', missing)]:
        (DEST / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    # Verify exported formats and unchanged numerical tables/images; never run analysis scripts here.
    for p in DEST.rglob('*'):
        if p.suffix == '.json':
            json.loads(p.read_text())
        elif p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p.relative_to(DEST)))
    for row in source_manifest:
        if Path(row['path']).suffix in {'.csv', '.png', '.svg', '.jpg'}:
            assert row['source_sha256'] == row['exported_sha256'], row['path']
    for p in DEST.rglob('*.md'):
        for m in LINK.finditer(p.read_text()):
            target = m.group(2).strip('<>').split('#')[0]
            if not target or re.match(r'[a-zA-Z]+:', target):
                continue
            assert (p.parent / urllib.parse.unquote(target)).exists(), (p, target)
    payload = {p.relative_to(DEST).as_posix(): sha(p.read_bytes()) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST / 'SHA256SUMS.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='prepared_not_uploaded', target='huzhe3566/foodLLM', visibility='private',
                   files=len(payload)+1, bytes=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),
                   source_files=len(source_manifest), transformed_files=sum(bool(r['transformations']) for r in source_manifest),
                   annotated_omitted_links=len(missing),
                   source_git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   source_includes_uncommitted_work=True, numerical_tables_and_images_unchanged=True,
                   json_and_python_syntax_valid=True, markdown_file_targets_valid=True)
    (DEST.parent / 'preparation_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
