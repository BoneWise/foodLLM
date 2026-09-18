"""Complete the descriptive regional report using immutable, already computed scores."""
import csv
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RSA = ROOT / 'outputs/20260918_控制RSA_v01'
OLD = ROOT / 'outputs/20260918_控制实验报告_v01'
V2 = ROOT / 'outputs/20260918_控制实验报告_v02'
OUT = ROOT / 'outputs/20260918_控制实验报告_v03'
CORE = ['OFC_nine', 'vmPFC', 'lOFC', 'mOFC', 'early_intermediate_visual', 'higher_visual']
DISPLAY = ['OFC 9-parcel summary', 'vmPFC (legacy sphere)', 'lOFC (legacy AAL)',
           'mOFC (legacy AAL)', 'Early/intermediate visual', 'Higher visual']
NUT_LABELS = ['Protein', 'Fat', 'Carbohydrate', 'Fibre']
CORDER = [2, 0, 1, 3]
CLABELS = ['T - V', 'N - V', 'T - mean N', 'T - mean P']
COLORS = ['#667085', '#8c4ca8', '#dc9009', '#168378']


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def readcsv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def writecsv(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def mdtable(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '| ' + ' | '.join(['---'] * len(head)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def direction(a, b):
    if a > 0 and b > 0: return 'both_positive'
    if a < 0 and b < 0: return 'both_negative'
    if a == 0 or b == 0: return 'includes_zero'
    return 'opposite_sign'


def main():
    OUT.mkdir(exist_ok=False); (OUT / 'figures').mkdir()
    protected = [p for d in [RSA, OLD, V2] for p in d.rglob('*') if p.is_file()]
    protected += [ROOT / 'configs/controls_v09_execution_v01.json',
                  ROOT / 'docs/research/20260918_无关词与随机分区健壮性设计_v09.md',
                  ROOT / 'docs/research/20260918_区域控制对照口径修订_v09_1.md']
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    ix = json.loads((RSA / 'index.json').read_text())
    names, nuts, prs = ix['series'], ix['nutrients'], ix['prompts']
    subjects = ix['subject_ids']; random = ix['random_parcels']
    vtp = np.load(RSA / 'series_VTP_scores.npy')
    ns = np.load(RSA / 'series_N_scores.npy')
    co = np.load(RSA / 'series_contrasts.npy')
    assert vtp.shape == (2, 4, 46, 29, 32, 7) and ns.shape == (2, 46, 29, 32, 3)
    assert co.shape == (2, 4, 46, 29, 32, 4)
    assert all(np.isfinite(x).all() for x in [vtp, ns, co])
    nmean = ns.mean(-1)[:, None]
    reconstructed = np.stack(np.broadcast_arrays(nmean-vtp[..., 0], vtp[..., 1]-nmean,
                                                 vtp[..., 1]-vtp[..., 0],
                                                 vtp[..., 1]-vtp[..., 2:].mean(-1)), axis=-1)
    contrast_error = float(np.max(np.abs(reconstructed-co)))
    assert contrast_error < 1e-12
    old_summary = {(r['prompt'], r['nutrient'], r['ROI'], r['window'], r['contrast']): r
                   for r in readcsv(RSA / 'window_summary.csv')}
    old_individual = {(r['prompt'], r['nutrient'], r['ROI'], r['window'], r['contrast'], r['subject_id']): float(r['value'])
                      for r in readcsv(RSA / 'participant_window_summary.csv')}
    ranges_old = {(r['prompt'], r['nutrient'], r['ROI'], r['window']): r
                  for r in readcsv(OLD / 'implementation_ranges.csv')}
    abs_old = {(r['prompt'], r['nutrient'], r['ROI'], int(r['layer'])): r
               for r in readcsv(RSA / 'absolute_curves.csv')}
    summaries, ab, implementations, random_summary = [], [], [], []
    win = {}; max_summary_error = 0.; max_individual_error = 0.; max_range_error = 0.; max_absolute_error = 0.
    for w, ls in ix['windows'].items():
        inds = np.array(ls)-1
        x = co[:, :, :, :, inds, :].mean(4)
        win[w] = x
        for pi, pr in enumerate(prs):
            for nj, nut in enumerate(nuts):
                for ri, roi in enumerate(names):
                    for ci, c in enumerate(ix['contrasts']):
                        y = x[pi, nj, ri, :, ci]; key = (pr, nut, roi, w, c)
                        old = old_summary[key]
                        vals = dict(mean=float(y.mean()), median=float(np.median(y)),
                                    subject_q25=float(np.quantile(y, .25)), subject_q75=float(np.quantile(y, .75)))
                        max_summary_error = max(max_summary_error, *(abs(vals[k]-float(old[k])) for k in vals))
                        assert int((y>0).sum()) == int(old['subjects_positive'])
                        for si, sub in enumerate(subjects):
                            max_individual_error = max(max_individual_error, abs(float(y[si])-old_individual[key+(sub,)]))
                        summaries.append(dict(prompt=pr, nutrient=nut, ROI=roi, window=w, contrast=c,
                                              **vals, positive=int((y>0).sum()), negative=int((y<0).sum()), zero=int((y==0).sum()), participants=29))
                    # Participant and layer means first; each N/P implementation remains separate.
                    n = ns[pi, ri][:, inds, :].mean((0, 1))
                    s = vtp[pi, nj, ri][:, inds, :].mean((0, 1))
                    dn, dp = s[1]-n, s[1]-s[2:]
                    old = ranges_old[(pr, nut, roi, w)]
                    check = dict(T_minus_N_min=dn.min(), T_minus_N_max=dn.max(), T_minus_P_min=dp.min(), T_minus_P_max=dp.max())
                    max_range_error = max(max_range_error, *(abs(float(v)-float(old[k])) for k,v in check.items()))
                    implementations.append(dict(prompt=pr, nutrient=nut, ROI=roi, window=w,
                        **{f'T_minus_N{i+1}':float(v) for i,v in enumerate(dn)},
                        **{f'T_minus_P{i+1}':float(v) for i,v in enumerate(dp)},
                        N_positive=int((dn>0).sum()), P_positive=int((dp>0).sum()),
                        N_range_min=float(dn.min()), N_range_max=float(dn.max()),
                        P_range_min=float(dp.min()), P_range_max=float(dp.max())))
        for nj,nut in enumerate(nuts):
            for ri,roi in enumerate(names):
                for ci,c in enumerate(ix['contrasts']):
                    a,b = x[0,nj,ri,:,ci],x[1,nj,ri,:,ci]
                    layera,layerb = co[:,nj,ri,:,:,ci].mean(1)[:,inds]
                    ab.append(dict(nutrient=nut, ROI=roi, window=w, contrast=c,
                        A_mean=float(a.mean()), B_mean=float(b.mean()), mean_direction=direction(a.mean(),b.mean()),
                        A_positive_subjects=int((a>0).sum()), B_positive_subjects=int((b>0).sum()),
                        paired_subjects_both_positive=int(((a>0)&(b>0)).sum()),
                        paired_subjects_both_negative=int(((a<0)&(b<0)).sum()),
                        paired_subjects_opposite_sign=int(((a*b)<0).sum()),
                        paired_subjects_includes_zero=int(((a==0)|(b==0)).sum()),
                        layers_same_nonzero_sign=int(((layera*layerb)>0).sum()),
                        layers_opposite_sign=int(((layera*layerb)<0).sum()), layers_in_window=len(inds)))
            for ci,c in enumerate(ix['contrasts']):
                aa=x[0,nj,[names.index(r) for r in random],:,ci].mean(1)
                bb=x[1,nj,[names.index(r) for r in random],:,ci].mean(1)
                for pi,pr in enumerate(prs):
                    z=[aa,bb][pi]
                    random_summary.append(dict(nutrient=nut, prompt=pr, window=w, contrast=c,
                        regions=20, positive=int((z>0).sum()), negative=int((z<0).sum()), zero=int((z==0).sum()),
                        median=float(np.median(z)), minimum=float(z.min()), maximum=float(z.max()),
                        AB_both_positive=int(((aa>0)&(bb>0)).sum()), AB_both_negative=int(((aa<0)&(bb<0)).sum()),
                        AB_opposite_sign=int((aa*bb<0).sum())))
    for pi,pr in enumerate(prs):
        for nj,nut in enumerate(nuts):
            for ri,roi in enumerate(names):
                s=vtp[pi,nj,ri].mean(0); n=ns[pi,ri].mean(0)
                for li in range(32):
                    values={'V':s[li,0],'T':s[li,1],'P_mean':s[li,2:].mean(),'N_mean':n[li].mean()}
                    old=abs_old[(pr,nut,roi,li+1)]
                    max_absolute_error=max(max_absolute_error,*(abs(float(v)-float(old[k])) for k,v in values.items()))
    assert max(max_summary_error,max_individual_error,max_range_error,max_absolute_error)<1e-12
    for n,r in [('participant_summary.csv',summaries),('AB_consistency.csv',ab),
                ('implementation_consistency.csv',implementations),('random_region_summary.csv',random_summary)]: writecsv(n,r)
    shutil.copy2(OLD/'ROI_quality_summary.csv',OUT/'ROI_quality_summary.csv')
    shutil.copy2(V2/'parallel_region_summary.csv',OUT/'parallel_region_summary.csv')
    shutil.copy2(V2/'全部区域并列摘要.md',OUT/'全部区域并列摘要.md')
    print('Numerical summaries and cross-checks complete',flush=True)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    figures=[]; layers=np.arange(1,33); corei=[names.index(r) for r in CORE]
    def save(fig,name,title):
        fig.suptitle(title,fontsize=12);fig.tight_layout(rect=(0,.02,1,.95))
        for ext in ['png','svg']: fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=120,bbox_inches='tight')
        plt.close(fig);figures.append(name)
    def basic(ax):
        ax.axhline(0,color='#888',lw=.65);ax.set_xlim(1,32);ax.set_xticks([1,8,16,24,32]);ax.set_xlabel('Layer')
    # Same absolute scale within each ROI across nutrients and prompts; x1000 throughout.
    for nj,label in enumerate(NUT_LABELS):
        fig,axs=plt.subplots(6,2,figsize=(12,16))
        for row,ri in enumerate(corei):
            bounds=[]
            for pi,pr in enumerate(prs):
                ax=axs[row,pi];s=vtp[pi,nj,ri].mean(0)*1000;n=ns[pi,ri].mean(0)*1000
                ys=[s[:,0],s[:,1],n.mean(1),s[:,2:].mean(1)]
                for y,c,l in zip(ys,COLORS,['V','T','mean N','mean P']):ax.plot(layers,y,color=c,label=l,lw=1.2)
                ax.fill_between(layers,n.min(1),n.max(1),color=COLORS[2],alpha=.16)
                ax.fill_between(layers,s[:,2:].min(1),s[:,2:].max(1),color=COLORS[3],alpha=.12)
                basic(ax);ax.set_title(DISPLAY[row]+' | '+pr);ax.set_ylabel('RSA x 1,000')
                bounds.extend([s.min(),s.max(),n.min(),n.max(),0])
            lo,hi=min(bounds),max(bounds);pad=max((hi-lo)*.08,.01)
            for ax in axs[row]:ax.set_ylim(lo-pad,hi+pad)
        axs[0,0].legend(ncol=4,fontsize=7)
        save(fig,f'absolute_{nj}',f'{label} | Parallel regional absolute RSA\n29 participant means; shading = N/P implementation range, not confidence intervals')
        fig,axs=plt.subplots(6,4,figsize=(17,17))
        for row,ri in enumerate(corei):
            for col,ci in enumerate(CORDER):
                ax=axs[row,col]
                for pi,pr in enumerate(prs):ax.plot(layers,co[pi,nj,ri,:,:,ci].mean(0)*1000,label=pr,color=['#196b9c','#df7642'][pi],ls=['-','--'][pi])
                basic(ax);ax.set_title(DISPLAY[row]+'\n'+CLABELS[col]);ax.set_ylabel('Delta RSA x 1,000')
        axs[0,0].legend()
        save(fig,f'contrasts_{nj}',f'{label} | All four contrasts, A and B\nShared food pairs; regional panels do not assume a preferred direction')
    # Full all41 heatmaps: common limits across prompts and nutrients for each contrast.
    means=co[:,:,:41].mean(3)*1000
    for pi,pr in enumerate(prs):
        fig,axs=plt.subplots(4,4,figsize=(19,25))
        for row,ci in enumerate(CORDER):
            bound=np.abs(means[:,:,:,:,ci]).max()
            for nj,label in enumerate(NUT_LABELS):
                ax=axs[row,nj];im=ax.imshow(means[pi,nj,:,:,ci],aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
                ax.set_yticks(range(41),names[:41],fontsize=6);ax.set_xticks([0,15,31],[1,16,32]);ax.set_title(label+' | '+CLABELS[row]);ax.set_xlabel('Layer')
                ax.axhline(17.5,color='#333',lw=.6);ax.axhline(37.5,color='#333',lw=.6)
                fig.colorbar(im,ax=ax,fraction=.028,pad=.02)
        save(fig,f'all41_{pr}',f'All 41 regions | Prompt {pr} | Delta RSA x 1,000\nEach contrast uses the same color limits across A/B and all nutrients; no significance thresholds')
    # Individual values: same participant color per prompt, deterministic horizontal offsets.
    for w,x in win.items():
        for nj,label in enumerate(NUT_LABELS):
            fig,axs=plt.subplots(6,4,figsize=(17,17))
            for row,ri in enumerate(corei):
                for col,ci in enumerate(CORDER):
                    ax=axs[row,col]
                    for pi,pr in enumerate(prs):
                        y=x[pi,nj,ri,:,ci]*1000
                        ax.scatter(pi+np.linspace(-.15,.15,29),y,s=12,color=['#196b9c','#df7642'][pi],alpha=.75)
                        ax.plot([pi-.22,pi+.22],[y.mean()]*2,color='black',lw=1.5)
                        ax.plot([pi-.17,pi+.17],[np.median(y)]*2,color='#6a2f88',lw=1.1,ls='--')
                    ax.axhline(0,color='gray',lw=.65);ax.set_xticks([0,1],prs);ax.set_title(DISPLAY[row]+'\n'+CLABELS[col]);ax.set_ylabel('Delta RSA x 1,000')
            save(fig,f'participants_{nj}_{w}',f'{label} | {w} | Paired participants, n=29\nBlack = mean; purple dashed = median; dots = participant window means; no confidence intervals')
    # Random panel: all 20, original draw order; includes both windows.
    for nj,label in enumerate(NUT_LABELS):
        fig,axs=plt.subplots(2,4,figsize=(18,14))
        for row,(w,x) in enumerate(win.items()):
            for col,ci in enumerate(CORDER):
                ax=axs[row,col]
                for pi,pr in enumerate(prs):
                    z=x[pi,nj,[names.index(r) for r in random],:,ci].mean(1)*1000
                    ax.scatter(z,np.arange(20)+[-.12,.12][pi],label=pr,color=['#196b9c','#df7642'][pi],s=23)
                ax.set_yticks(range(20),random);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.7);ax.set_title(w+' | '+CLABELS[col]);ax.set_xlabel('Delta RSA x 1,000')
        axs[0,0].legend()
        save(fig,f'random_{nj}',f'{label} | All 20 fixed random regions\nPer-region participant means; anatomical references, not independent replicates or a null distribution')
    # Atlas guarantees every existing series has complete absolute and contrast curves.
    atlas=['# 逐区域完整曲线索引','41套区域及5组既有汇总均保留，每页含四营养、A/B绝对分数与四项差值。阴影为N/P实现范围，不是置信区间。全32层，没有按效果选择层或分区。','[返回结果报告](结果报告.md)']
    for ri,roi in enumerate(names):
        fig,axs=plt.subplots(4,4,figsize=(18,12))
        for nj,label in enumerate(NUT_LABELS):
            for pi,pr in enumerate(prs):
                s=vtp[pi,nj,ri].mean(0)*1000;n=ns[pi,ri].mean(0)*1000
                ax=axs[nj,pi]
                for y,c,l in zip([s[:,0],s[:,1],n.mean(1),s[:,2:].mean(1)],COLORS,['V','T','mean N','mean P']):ax.plot(layers,y,color=c,label=l,lw=1)
                ax.fill_between(layers,n.min(1),n.max(1),color=COLORS[2],alpha=.13);ax.fill_between(layers,s[:,2:].min(1),s[:,2:].max(1),color=COLORS[3],alpha=.13)
                ax.set_title(label+' | '+pr+' absolute');basic(ax)
                ax=axs[nj,pi+2]
                for ci,c,l in zip(CORDER,COLORS,CLABELS):ax.plot(layers,co[pi,nj,ri,:,:,ci].mean(0)*1000,color=c,label=l,lw=1)
                ax.set_title(label+' | '+pr+' contrasts');basic(ax)
            for columns in [(0,1),(2,3)]:
                lim=[axs[nj,c].get_ylim() for c in columns];lo=min(l[0] for l in lim);hi=max(l[1] for l in lim)
                for c in columns:axs[nj,c].set_ylim(lo,hi)
        axs[0,0].legend(ncol=4,fontsize=6);axs[0,2].legend(ncol=2,fontsize=6)
        fname=f'region_{roi}';save(fig,fname,f'{roi} | Complete regional profile | RSA / Delta RSA x 1,000\nA/B matched axis scales per nutrient; N/P shading is implementation spread')
        atlas.append(f'- {roi}：[PNG](figures/{fname}.png) · [SVG](figures/{fname}.svg)')
        if ri%10==0:print(f'Atlas {ri+1}/46',flush=True)
    (OUT/'逐区域曲线索引.md').write_text('\n\n'.join(atlas)+'\n')
    # QA contact sheets (read all pages before delivery).
    for page,start in enumerate(range(0,len(figures),12),1):
        batch=figures[start:start+12];sheet=Image.new('RGB',(1600,4*390),'white');draw=ImageDraw.Draw(sheet)
        for j,name in enumerate(batch):
            im=Image.open(OUT/'figures'/f'{name}.png').convert('RGB');im.thumbnail((390,355))
            xx=(j%4)*400;yy=(j//4)*390
            sheet.paste(im,(xx+(390-im.width)//2,yy+22));draw.text((xx+8,yy+4),name,fill='black')
        sheet.save(OUT/f'contact_sheet_{page}.jpg',quality=88)
    assert all(sha(ROOT/p)==v for p,v in inputs.items())
    shutil.copy2(__file__,OUT/'script_snapshot.py')
    audit=dict(status='numerical_and_plot_generation_complete_pending_report_and_visual_QA',
        created=datetime.now().astimezone().isoformat(),
        user_scope='Complete missing descriptive results, plots and interpretation after regional-control reframing.',
        new_model_inference=False,new_brain_extraction=False,RSA_recomputed=False,inference_run=False,
        random_seed=None,random_seed_reason='No random sampling; point offsets are deterministic.',
        parameters=dict(core_regions=CORE,all_series=names,windows=ix['windows'],display_scale=1000,
                        direction_rule='Strict signs of unrounded saved means; no practical-equivalence threshold.',
                        no_effect_selection=True),
        input_sha256=inputs,protected_files_unchanged=True,
        validation=dict(contrast_identity_max_error=contrast_error,window_summary_max_error=max_summary_error,
                        individual_window_max_error=max_individual_error,implementation_ranges_max_error=max_range_error,
                        absolute_curve_max_error=max_absolute_error,participant_summary_rows=len(summaries),
                        AB_rows=len(ab),implementation_rows=len(implementations),random_summary_rows=len(random_summary)),
        figures=figures,git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status_note='Pre-existing local changes retained; exact generation script snapshotted; no remote push.',
        dependencies=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
    (OUT/'generation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figures':len(figures),'validation':audit['validation']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
