"""V-only standardized ridge with explicit food-group cross-fitting.

These small functions return training-group traces for leakage auditing.
They do not implement population confidence intervals.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass
class Readout:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    intercept: float
    regularization: float
    trained_groups: frozenset

    def predict(self, x):
        return ((np.asarray(x) - self.mean) / self.scale) @ self.weight + self.intercept


def fit_at_lambda(x, y, groups, regularization):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.
    z = (x - mean) / scale
    intercept = float(y.mean())
    alpha = len(y) * regularization
    if z.shape[1] <= z.shape[0]:
        w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    else:
        w = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(z.shape[0]), y - intercept)
    return Readout(mean, scale, w, intercept, regularization, frozenset(groups))


def tune_v(x_v, y, groups, indices, lambdas, inner_folds=3):
    indices = np.asarray(indices, dtype=int)
    group_array = np.asarray(groups)
    folds = list(GroupKFold(inner_folds).split(indices, groups=group_array[indices]))
    losses = []
    for lam in lambdas:
        sse = 0.
        count = 0
        for train, valid in folds:
            a, b = indices[train], indices[valid]
            assert set(group_array[a]).isdisjoint(group_array[b])
            model = fit_at_lambda(x_v[a], y[a], group_array[a], lam)
            sse += float(np.sum((model.predict(x_v[b]) - y[b]) ** 2))
            count += len(b)
        losses.append(sse / count)
    # Exact tie: larger lambda. No test condition or test labels are consulted.
    chosen = min(range(len(lambdas)), key=lambda i: (losses[i], -lambdas[i]))
    return fit_at_lambda(x_v[indices], y[indices], group_array[indices], lambdas[chosen])


def crossfit_v_conditions(features, y, groups, train_indices, test_indices,
                         lambdas=(1e-3, 1e-1, 10.), crossfit_folds=4, inner_folds=3):
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    assert set(groups[train_indices]).isdisjoint(groups[test_indices])
    output = {c: np.full(len(y), np.nan) for c in features}
    traces = []
    for a, j in GroupKFold(crossfit_folds).split(train_indices, groups=groups[train_indices]):
        fit_ids, held_ids = train_indices[a], train_indices[j]
        model = tune_v(features['V'], y, groups, fit_ids, lambdas, inner_folds)
        assert model.trained_groups.isdisjoint(groups[held_ids])
        assert model.trained_groups.isdisjoint(groups[test_indices])
        for c, x in features.items():
            output[c][held_ids] = model.predict(x[held_ids])
        traces.append({'fit_indices': fit_ids.tolist(), 'held_indices': held_ids.tolist(),
                       'lambda': model.regularization})
    final = tune_v(features['V'], y, groups, train_indices, lambdas, inner_folds)
    for c, x in features.items():
        output[c][test_indices] = final.predict(x[test_indices])
    return output, traces, final


def mean_replica_mse(predictions, target):
    return float(np.mean([(np.asarray(p) - target) ** 2 for p in predictions]))
