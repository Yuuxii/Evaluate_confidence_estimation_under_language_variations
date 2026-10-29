"""Language-variation metrics.

* P-RB  (prompt robustness): confidence spread across prompt paraphrases that
  yield semantically equivalent greedy answers.
* A-STB (answer stability): confidence spread within the largest group of
  semantically equivalent (but textually distinct) sampled answers.
* A-SST (answer sensitivity): how much more confidence differs between
  semantically different sampled answers than within the largest group.
"""

import os

import numpy as np
import pandas as pd

from .data import (
    greedy_groups_path,
    groups_path,
    numeric_series,
    parse_group_indices,
    parse_prompt_index,
    read_csv,
)
from .metrics import mean_gap, nanmean_or_nan, row_score


# ------------------------------------------------------- prompt robustness

def group_containing_ans0(groups):
    for group in groups:
        if 0 in group:
            return group
    return []


def prompt_robustness_values(args, dataname, model, greedy_frames, method):
    """Return (mean P-RB, per-question P-RB values, auxiliary stats)."""
    group_path = greedy_groups_path(args, dataname, model)
    if not os.path.isfile(group_path):
        print(f"Skipping robustness for {dataname}/{model}: missing greedy groups file {group_path}")
        return np.nan, None, {}
    group_frame = read_csv(group_path)
    if "semantic_group_indices" not in group_frame.columns:
        print(f"Skipping robustness for {dataname}/{model}: greedy groups file lacks semantic_group_indices")
        return np.nan, None, {}

    method_scores = []
    prompt_to_pos = {}
    for prompt_name, frame in greedy_frames:
        scores = numeric_series(frame, method)
        if scores is not None:
            prompt_idx = parse_prompt_index(prompt_name)
            if prompt_idx is not None:
                prompt_to_pos[prompt_idx] = len(method_scores)
            method_scores.append(scores.reset_index(drop=True))
    if len(method_scores) < 2:
        return np.nan, None, {}

    values = [np.nan] * len(group_frame)
    surviving_prompt_fractions = [np.nan] * len(group_frame)
    all_agree_values = [np.nan] * len(group_frame)
    partial_agree_values = [np.nan] * len(group_frame)
    prompt_count = args.num_answer_prompts
    if prompt_count == 0:
        return np.nan, values, {}

    n = min(len(group_frame), *(len(scores) for scores in method_scores))
    for row_idx in range(n):
        groups = parse_group_indices(group_frame.at[row_idx, "semantic_group_indices"])
        ans0_group = group_containing_ans0(groups)
        if ans0_group:
            surviving_prompt_fractions[row_idx] = len(set(ans0_group)) / prompt_count
        if len(ans0_group) < 2:
            continue
        row_scores = [
            method_scores[prompt_to_pos[prompt_idx]].iloc[row_idx]
            for prompt_idx in ans0_group
            if prompt_idx in prompt_to_pos and pd.notna(method_scores[prompt_to_pos[prompt_idx]].iloc[row_idx])
        ]
        value = row_score(row_scores, invert_std=True)
        values[row_idx] = value
        if pd.notna(value):
            if len(set(ans0_group)) == prompt_count:
                all_agree_values[row_idx] = value
            else:
                partial_agree_values[row_idx] = value

    valid_surviving = [v for v in surviving_prompt_fractions if pd.notna(v)]
    stats = {
        "surviving_prompt_fraction": nanmean_or_nan(valid_surviving),
        "all_agree_fraction": float(np.nanmean([v == 1.0 for v in valid_surviving])) if valid_surviving else np.nan,
        "partial_agree_fraction": float(np.nanmean([0.0 < v < 1.0 for v in valid_surviving]))
        if valid_surviving
        else np.nan,
        "robustness_all_agree": nanmean_or_nan(all_agree_values),
        "robustness_partial_agree": nanmean_or_nan(partial_agree_values),
        "surviving_prompt_fraction_values": surviving_prompt_fractions,
        "robustness_all_agree_values": all_agree_values,
        "robustness_partial_agree_values": partial_agree_values,
    }
    return nanmean_or_nan(values), values, stats


# ------------------------------------------- answer stability / sensitivity

def sample_values(sample_frames, method, row_idx):
    values = {}
    for sample_idx, frame in sample_frames:
        scores = numeric_series(frame, method, sample=True)
        if scores is not None and row_idx < len(scores) and pd.notna(scores.iloc[row_idx]):
            values[sample_idx] = float(scores.iloc[row_idx])
    return values


def clean_answer_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def sample_answers(sample_frames, row_idx):
    answers = {}
    for sample_idx, frame in sample_frames:
        if "response" in frame.columns and row_idx < len(frame):
            answers[sample_idx] = clean_answer_text(frame.at[row_idx, "response"])
    return answers


def has_distinct_answer_text(group, answers):
    texts = {answers[i] for i in group if i in answers and answers[i] != ""}
    return len(texts) >= 2


def largest_semantic_nonexact_group(groups, answers, values):
    """Largest semantic group containing at least two textually distinct answers."""
    candidates = []
    for group in groups:
        valid_group = [i for i in group if i in values]
        if len(valid_group) >= 2 and has_distinct_answer_text(valid_group, answers):
            candidates.append(valid_group)
    if not candidates:
        return []
    return max(candidates, key=len)


def sample_stability_sensitivity_values(args, dataname, model, sample_frames, method):
    """Return (mean A-STB, mean A-SST, per-question A-STB, per-question A-SST)."""
    if len(sample_frames) < 2:
        return np.nan, np.nan, None, None

    group_file = groups_path(args, dataname, model)
    group_frame = read_csv(group_file) if os.path.isfile(group_file) else None
    sample_len = min(len(frame) for _, frame in sample_frames)
    output_len = len(group_frame) if group_frame is not None else sample_len
    if group_frame is not None and output_len < sample_len:
        print(
            f"Warning: sampled group file has fewer rows than sampled CSVs: {group_file} "
            f"(group rows={output_len}, sample rows={sample_len})."
        )
    n = min(output_len, sample_len)
    stability_values = [np.nan] * output_len
    sensitivity_values = [np.nan] * output_len

    for row_idx in range(n):
        values = sample_values(sample_frames, method, row_idx)
        if len(values) < 2:
            continue
        answers = sample_answers(sample_frames, row_idx)

        if group_frame is not None and row_idx < len(group_frame) and "semantic_group_indices" in group_frame.columns:
            groups = parse_group_indices(group_frame.at[row_idx, "semantic_group_indices"])
        else:
            groups = [list(values.keys())]

        group_scores = [[values[i] for i in group if i in values] for group in groups]
        group_scores = [scores for scores in group_scores if scores]
        if not group_scores:
            continue

        largest_index = max(range(len(group_scores)), key=lambda idx: len(group_scores[idx]))
        smallest_index = min(range(len(group_scores)), key=lambda idx: len(group_scores[idx]))
        largest = group_scores[largest_index]
        smallest = group_scores[smallest_index]

        stability_group = largest_semantic_nonexact_group(groups, answers, values)
        if stability_group:
            stability_values[row_idx] = row_score([values[i] for i in stability_group], invert_std=True)
        if largest_index != smallest_index and len(largest) >= 2 and smallest:
            sensitivity_values[row_idx] = mean_gap(largest, smallest)

    return (
        nanmean_or_nan(stability_values),
        nanmean_or_nan(sensitivity_values),
        stability_values,
        sensitivity_values,
    )
