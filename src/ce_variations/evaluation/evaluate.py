"""Per-(dataset, model) evaluation of every CE method."""

import os

import numpy as np
import pandas as pd

from .data import (
    csv_path,
    greedy_groups_path,
    groups_path,
    load_greedy_frames,
    load_sample_frames,
    load_supervised_ans0,
    method_col_prefix,
    read_csv,
    save_group_metric_columns,
)
from .config import REFERENCE_PROMPT
from .metrics import calibration_metrics
from .variation import prompt_robustness_values, sample_stability_sensitivity_values


def coverage_text(name, values):
    if values is None:
        return f"{name}: 0/0 (nan%)"
    used, total = sum(pd.notna(v) for v in values), len(values)
    if not total:
        return f"{name}: 0/0 (nan%)"
    return f"{name}: {used}/{total} ({100 * used / total:.1f}%)"


def accuracy_one(args, dataname, model):
    """Accuracy of greedy reference-prompt answers according to the judge labels."""
    row = {"dataset": dataname, "model": model, "accuracy": np.nan, "num_examples": 0}
    label_path = csv_path(args, dataname, model, "test", "greedy", REFERENCE_PROMPT, with_gpt_score=True)
    if not os.path.isfile(label_path):
        print(f"Skipping accuracy for {dataname}/{model}: missing {label_path}")
        return row
    frame = read_csv(label_path)
    if args.label_col not in frame.columns:
        print(f"Skipping accuracy for {dataname}/{model}: {label_path} lacks {args.label_col}")
        return row
    labels = pd.to_numeric(frame[args.label_col], errors="coerce")
    labels = labels[labels.notna() & (labels != -1)]
    if len(labels):
        row["accuracy"] = float(labels.mean())
        row["num_examples"] = int(len(labels))
    return row


def evaluate_one(args, dataname, model, accuracy=np.nan):
    greedy_frames = load_greedy_frames(args, dataname, model)
    sample_frames = load_sample_frames(args, dataname, model)
    if not greedy_frames:
        return []

    ans0 = load_supervised_ans0(args, dataname, model)
    rows = []
    greedy_updates = {}
    sample_updates = {}
    for method in args.methods:
        ece, brier_score, auroc = calibration_metrics(ans0, method, args.label_col)
        robustness, robustness_values, rb_stats = prompt_robustness_values(
            args, dataname, model, greedy_frames, method
        )
        stability, sensitivity, stability_values, sensitivity_values = sample_stability_sensitivity_values(
            args, dataname, model, sample_frames, method
        )
        prefix = method_col_prefix(method)
        greedy_updates[f"{prefix}_robustness"] = robustness_values
        greedy_updates[f"{prefix}_surviving_prompt_fraction"] = rb_stats.get("surviving_prompt_fraction_values")
        greedy_updates[f"{prefix}_robustness_all_agree"] = rb_stats.get("robustness_all_agree_values")
        greedy_updates[f"{prefix}_robustness_partial_agree"] = rb_stats.get("robustness_partial_agree_values")
        sample_updates[f"{prefix}_stability"] = stability_values
        sample_updates[f"{prefix}_sensitivity"] = sensitivity_values
        print(
            f"Coverage {dataname}/{model}/{method}: "
            f"{coverage_text('P-RB', robustness_values)}, "
            f"{coverage_text('A-STB', stability_values)}, "
            f"{coverage_text('A-SST', sensitivity_values)}, "
            f"|I|/M={rb_stats.get('surviving_prompt_fraction', np.nan):.3f}, "
            f"P-RB_all={rb_stats.get('robustness_all_agree', np.nan):.3f}, "
            f"P-RB_partial={rb_stats.get('robustness_partial_agree', np.nan):.3f}"
        )
        rows.append(
            {
                "dataset": dataname,
                "model": model,
                "method": method,
                "accuracy": accuracy,
                "ece": ece,
                "brier": brier_score,
                "auroc": auroc,
                "robustness": robustness,
                "surviving_prompt_fraction": rb_stats.get("surviving_prompt_fraction", np.nan),
                "all_agree_fraction": rb_stats.get("all_agree_fraction", np.nan),
                "partial_agree_fraction": rb_stats.get("partial_agree_fraction", np.nan),
                "robustness_all_agree": rb_stats.get("robustness_all_agree", np.nan),
                "robustness_partial_agree": rb_stats.get("robustness_partial_agree", np.nan),
                "stability": stability,
                "sensitivity": sensitivity,
            }
        )
    if args.save_group_scores:
        save_group_metric_columns(greedy_groups_path(args, dataname, model), greedy_updates)
        save_group_metric_columns(groups_path(args, dataname, model), sample_updates)
    return rows
