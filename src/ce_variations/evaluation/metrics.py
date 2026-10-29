"""Standard CE metrics (ECE, Brier, AUROC) and small numeric helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import numeric_series, valid_xy


def compute_ece(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))
    return float(ece)


def compute_auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    binary = (labels >= 0.5).astype(int)
    if len(np.unique(binary)) < 2:
        return np.nan
    return float(roc_auc_score(binary, scores))


def compute_brier(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) == 0:
        return np.nan
    return float(np.mean((labels - scores) ** 2))


def score_std(values):
    values = [x for x in values if pd.notna(x)]
    if len(values) < 2:
        return np.nan
    return float(np.std(values))


def row_score(values, invert_std=False):
    """Spread of confidences within one group; `1 - std` when `invert_std`."""
    row_std = score_std(values)
    if pd.isna(row_std):
        return np.nan
    return 1 - row_std if invert_std else row_std


def mean_gap(list1, list2):
    """|mean cross-group gap - mean within-group gap| between two score lists."""
    list1 = [x for x in list1 if pd.notna(x)]
    list2 = [x for x in list2 if pd.notna(x)]
    if len(list1) < 2 or not list2:
        return np.nan
    ingroup = [abs(x - y) for x in list1 for y in list1]
    outgroup = [abs(x - y) for x in list1 for y in list2]
    return float(abs(np.mean(outgroup) - np.mean(ingroup)))


def nanmean_or_nan(values):
    valid = [v for v in values if pd.notna(v)]
    return float(np.nanmean(valid)) if valid else np.nan


def calibration_metrics(ans0, method, label_col):
    """(ECE, Brier, AUROC) of `method` on the reference-prompt greedy answers."""
    nans = (np.nan, np.nan, np.nan)
    if ans0 is None or label_col not in ans0.columns:
        return nans
    scores = numeric_series(ans0, method)
    if scores is None:
        return nans
    labels, scores = valid_xy(pd.to_numeric(ans0[label_col], errors="coerce"), scores)
    return compute_ece(labels, scores), compute_brier(labels, scores), compute_auroc(labels, scores)
