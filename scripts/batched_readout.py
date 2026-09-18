"""Exactly the V-only ridge estimator, sharing linear algebra across human targets.

Each target selects its own lambda. Designs, scaling and splits are shared;
labels, lambda choices and intercepts are not pooled across targets.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.model_selection import GroupKFold


class RidgeDesign:
    def __init__(self, x):
        x = np.asarray(x, dtype=float)
        self.mean = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale < 1e-12] = 1.
        self.z = (x-self.mean)/self.scale
        self.n = len(x)
        self.eigenvalues, self.u = eigh(self.z @ self.z.T, check_finite=False)
        tolerance = max(float(self.eigenvalues[-1]), 1.) * 1e-12
        if self.eigenvalues[0] < -tolerance:
            raise ValueError('Unexpected non-positive-semidefinite Gram matrix')
        self.eigenvalues = np.maximum(self.eigenvalues, 0.)

    def validation_predictions(self, y, x, lambdas):
        y = np.asarray(y, dtype=float)
        intercept = y.mean(0)
        projected_y = self.u.T @ (y-intercept)
        cross = ((np.asarray(x, dtype=float)-self.mean)/self.scale) @ self.z.T @ self.u
        return np.stack([cross @ (projected_y/(self.eigenvalues[:, None]+self.n*lam))
                         + intercept for lam in lambdas])

    def fit_weights(self, y, lambdas):
        intercept = y.mean(0)
        dual = self.u @ ((self.u.T @ (y-intercept)) /
                        (self.eigenvalues[:, None]+self.n*np.asarray(lambdas)[None, :]))
        return self.z.T @ dual, intercept


def tune_predict(features, targets, groups, train, predict, lambdas, inner_folds):
    """Return per-condition (held rows, targets), independent target-wise lambdas."""
    train = np.asarray(train, dtype=int)
    predict = np.asarray(predict, dtype=int)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]
    groups = np.asarray(groups)
    assert set(groups[train]).isdisjoint(groups[predict])
    losses = np.zeros((len(lambdas), targets.shape[1]))
    n_valid = 0
    for a, b in GroupKFold(inner_folds).split(train, groups=groups[train]):
        fit, held = train[a], train[b]
        assert set(groups[fit]).isdisjoint(groups[held])
        design = RidgeDesign(features['V'][fit])
        pred = design.validation_predictions(targets[fit], features['V'][held], lambdas)
        losses += ((pred-targets[held][None, :, :])**2).sum(1)
        n_valid += len(held)
    losses /= n_valid
    # Same exact tie rule as the reference implementation: largest lambda.
    chosen = np.array([min(range(len(lambdas)), key=lambda k: (losses[k,j],-lambdas[k]))
                       for j in range(targets.shape[1])])
    selected = np.asarray(lambdas)[chosen]
    design = RidgeDesign(features['V'][train])
    weights, intercept = design.fit_weights(targets[train], selected)
    result = {condition: ((np.asarray(x[predict],dtype=float)-design.mean)/design.scale)
              @ weights + intercept for condition,x in features.items()}
    return result, selected, losses


def crossfit_predict(features, targets, groups, train, test, lambdas,
                     crossfit_folds=4, inner_folds=3):
    n, s = targets.shape
    q = {condition: np.full((n,s),np.nan) for condition in features}
    traces = []
    for a,j in GroupKFold(crossfit_folds).split(train,groups=np.asarray(groups)[train]):
        fit,held = train[a],train[j]
        assert set(np.asarray(groups)[fit]).isdisjoint(np.asarray(groups)[test])
        result,selected,_ = tune_predict(features,targets,groups,fit,held,lambdas,inner_folds)
        for condition in features:
            q[condition][held] = result[condition]
        traces.append({'fit_indices':fit.tolist(),'held_indices':held.tolist(),
                       'lambda_by_target':selected.tolist()})
    result,selected,_ = tune_predict(features,targets,groups,train,test,lambdas,inner_folds)
    for condition in features:
        q[condition][test] = result[condition]
    traces.append({'fit_indices':train.tolist(),'held_indices':test.tolist(),
                   'lambda_by_target':selected.tolist(),'role':'outer_test'})
    assert all(np.isfinite(x).all() for x in q.values())
    return q,traces
