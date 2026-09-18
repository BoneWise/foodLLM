"""Fixed correlation-distance / Spearman operations; no inference or fitting."""
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import pdist, squareform

def correlation_rdm(patterns):
    x=np.asarray(patterns,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all(): raise ValueError('Non-finite or invalid patterns')
    z=x-x.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(z,axis=1,keepdims=True)
    if np.any(norm==0): raise ValueError('Constant pattern')
    z=z/norm; corr=z@z.T
    if np.max(np.abs(corr))>1+1e-12: raise ValueError('Correlation outside numerical tolerance')
    d=1-np.clip(corr,-1,1);np.fill_diagonal(d,0)
    if not np.allclose(d,d.T,atol=1e-14,rtol=0): raise ValueError('Asymmetric RDM')
    return d

def ranked_unit(v):
    v=np.asarray(v,dtype=np.float64)
    if not np.isfinite(v).all(): raise ValueError('Non-finite RDM entries')
    z=rankdata(v,method='average');z-=z.mean();n=np.linalg.norm(z)
    if n==0: raise ValueError('Constant RDM vector')
    return z/n

def score(a,b):
    return float(ranked_unit(a)@ranked_unit(b))

def validate():
    rng=np.random.default_rng(2026091701)
    x=rng.normal(size=(12,9));y=rng.normal(size=(12,7))
    x[2]=x[0];x[5]=x[3]  # ties remain intentional
    a=correlation_rdm(x); b=correlation_rdm(y);t=np.tril_indices(12,-1)
    error=float(np.max(np.abs(a-squareform(pdist(x,'correlation')))))
    assert error<1e-12
    assert abs(score(a[t],b[t])-spearmanr(a[t],b[t]).statistic)<1e-12
    assert abs(score([0,0,1,2,2],[2,1,1,0,0])-spearmanr([0,0,1,2,2],[2,1,1,0,0]).statistic)<1e-12
    perm=rng.permutation(12)
    assert abs(score(a[t],b[t])-score(a[np.ix_(perm,perm)][t],b[np.ix_(perm,perm)][t]))<1e-12
    keep=np.array([0,1,3,4,6,9]);tt=np.tril_indices(len(keep),-1)
    assert np.allclose(correlation_rdm(x[keep]),a[np.ix_(keep,keep)],atol=1e-12)
    direct=score(a[np.ix_(keep,keep)][tt],b[np.ix_(keep,keep)][tt])
    assert abs(direct-spearmanr(a[keep[tt[0]],keep[tt[1]]],b[keep[tt[0]],keep[tt[1]]]).statistic)<1e-12
    ps=[correlation_rdm(rng.normal(size=(12,9))) for _ in range(5)]
    scores=np.array([score(p[t],b[t]) for p in ps]); baseline=float(scores.mean())
    assert abs(baseline-score(np.mean(ps,axis=0)[t],b[t]))>1e-6
    for fn in [lambda:correlation_rdm(np.ones((3,4))),lambda:ranked_unit(np.ones(6)),lambda:ranked_unit([1,np.nan])]:
        try: fn()
        except ValueError: pass
        else: raise AssertionError('Undefined input accepted')
    return {'status':'passed','fixture_seed':2026091701,'pdist_max_error':error,'checks':['independent_Pearson_distance','Spearman_vs_scipy','average_ties','ID_reordering','shared_missing_mask','P_score_mean_distinct_from_RDM_mean','invalid_constant_or_NaN_rejected'],'p_values_computed':False}
