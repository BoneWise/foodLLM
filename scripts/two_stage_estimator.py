"""One-prompt, one-layer, one-subject neural fold with explicit target isolation.

This core does not implement population bootstrap or full research execution.
All arrays must already use the same ordered image index and frozen food folds.
"""
import numpy as np

from frozen_readout import crossfit_v_conditions


def leave_subject_out_mean(human_ratings, target_subject):
    human = np.asarray(human_ratings, dtype=float)
    others = np.delete(human, target_subject, axis=0)
    count = np.isfinite(others).sum(axis=0)
    if np.any(count == 0):
        raise ValueError('Every image needs at least one non-target human rating')
    return np.nansum(others, axis=0) / count


def fit_ols_predict(train_x, train_y, test_x):
    """Scale using training covariates only; intercept and OLS predictions are retained."""
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    assert np.isfinite(train_x).all() and np.isfinite(train_y).all() and np.isfinite(test_x).all()
    mean = train_x.mean(0)
    scale = train_x.std(0)
    scale[scale < 1e-12] = 1.
    train = np.column_stack([np.ones(len(train_x)), (train_x-mean)/scale])
    test = np.column_stack([np.ones(len(test_x)), (test_x-mean)/scale])
    weight, _, rank, _ = np.linalg.lstsq(train, train_y, rcond=None)
    return test @ weight, {'weight':weight, 'mean':mean, 'scale':scale, 'rank':int(rank)}


def neural_fold(features, human_ratings, target_subject, beta, content_covariates,
                raw_V_probs, food_groups, outer_train, outer_test, run_ids,
                session_ids=None, held_session=None,
                lambdas=tuple(10.**k for k in range(-6, 7))):
    """Fit the prespecified low-dimensional models for one outer fold.

    content_covariates must be red/green/blue/object_size/log1p(grams)/correct_kcal.
    Run identifiers must include session identity. held_session activates the
    food-by-session check with a training global intercept, never test centering.
    """
    groups = np.asarray(food_groups).astype(str)
    runs = np.asarray(run_ids).astype(str)
    beta = np.asarray(beta, dtype=float)
    content = np.asarray(content_covariates, dtype=float)
    probs = np.asarray(raw_V_probs, dtype=float)
    train = np.asarray(outer_train, dtype=int)
    test = np.asarray(outer_test, dtype=int)
    n = len(beta)
    assert content.shape == (n, 6) and probs.shape == (n, 4)
    assert np.allclose(probs.sum(1), 1., atol=1e-6)
    assert set(groups[train]).isdisjoint(groups[test])
    target = leave_subject_out_mean(human_ratings, target_subject)
    assert len(target) == n
    q, traces, readout = crossfit_v_conditions(features, target, groups, train, test,
        lambdas=lambdas, crossfit_folds=4, inner_folds=3)
    # Missing target beta affects only availability in this subject's brain fit.
    brain_train = train[np.isfinite(beta[train])]
    brain_test = test[np.isfinite(beta[test])]
    if held_session is not None:
        if session_ids is None:
            raise ValueError('Food-by-session evaluation requires session IDs')
        sessions = np.asarray(session_ids).astype(str)
        brain_train = brain_train[sessions[brain_train] != str(held_session)]
        brain_test = brain_test[sessions[brain_test] == str(held_session)]
    if not len(brain_train) or not len(brain_test):
        raise ValueError('Empty brain training or test fold')
    base = np.column_stack([content, probs[:, :3]])
    levels = []
    if held_session is None:
        levels = sorted(set(runs[brain_train]))
        assert set(runs[brain_test]).issubset(levels), 'Test run has no training-food observations'
        if len(levels) > 1:
            base = np.column_stack([base, *[(runs == level).astype(float) for level in levels[1:]]])
    pred, models, errors = {}, {}, {}
    for condition in ['baseline', *features]:
        x = base if condition == 'baseline' else np.column_stack([base, q[condition]])
        pred[condition], models[condition] = fit_ols_predict(
            x[brain_train], beta[brain_train], x[brain_test])
        errors[condition] = (pred[condition]-beta[brain_test])**2
    p_conditions = sorted(c for c in features if c.startswith('P'))
    if not p_conditions:
        raise ValueError('Main V/T/P comparison requires P realizations')
    return {'test_indices':brain_test, 'train_indices':brain_train, 'predictions':pred,
        'squared_errors':errors, 'G':errors['V']-errors['T'],
        'S':np.mean([errors[c] for c in p_conditions], axis=0)-errors['T'],
        'q':q, 'crossfit_traces':traces, 'V_readout':readout, 'brain_models':models,
        'run_levels':levels, 'held_session':held_session}
