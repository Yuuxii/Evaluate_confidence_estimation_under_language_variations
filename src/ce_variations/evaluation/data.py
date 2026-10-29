"""Reading response CSVs and group files produced by the inference pipeline."""

import json
import os
import re

import numpy as np
import pandas as pd

from .config import (
    METHOD_ALIASES,
    PROMPT_PREFIX,
    REFERENCE_PROMPT,
    SAMPLE_METHOD_ALIASES,
)


def read_csv(path):
    frame = pd.read_csv(path, encoding="utf-8")
    return frame.loc[:, ~frame.columns.str.contains("^Unnamed")]


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8")


def prompt_names(num_prompts):
    return [f"{PROMPT_PREFIX}{i}" for i in range(num_prompts)]


def parse_prompt_index(prompt_name):
    text = str(prompt_name)
    if text.startswith(PROMPT_PREFIX):
        try:
            return int(text[len(PROMPT_PREFIX):])
        except ValueError:
            return None
    return None


def method_col_prefix(method):
    prefix = re.sub(r"[^0-9a-zA-Z]+", "_", str(method).lower()).strip("_")
    return prefix or "method"


# ---------------------------------------------------------------- file paths

def csv_path(args, dataname, model, split, kind, prompt, sample_idx=None, with_gpt_score=False):
    suffix = f"{split}_{kind}_{prompt}"
    if sample_idx is not None:
        suffix += f"_{sample_idx}"
    if with_gpt_score:
        suffix += "_with_gpt_score"
    return os.path.join(args.response_dir, f"{dataname}_{model}_{suffix}.csv")


def groups_path(args, dataname, model):
    return os.path.join(args.response_dir, f"{dataname}_{model}_test_sampled_groups.csv")


def greedy_groups_path(args, dataname, model):
    return os.path.join(args.response_dir, f"{dataname}_{model}_test_greedy_groups.csv")


# ------------------------------------------------------------------- scores

def score_col(frame, method, sample=False):
    aliases = SAMPLE_METHOD_ALIASES if sample else METHOD_ALIASES
    for col in aliases[method]:
        if col in frame.columns:
            return col
    return None


def numeric_series(frame, method, sample=False):
    """Confidence scores of `method` as floats; values outside [0, 1] become NaN."""
    col = score_col(frame, method, sample=sample)
    if col is None:
        return None
    values = pd.to_numeric(frame[col], errors="coerce")
    return values.where((values >= 0) & (values <= 1))


def valid_xy(labels, scores):
    """Drop rows with missing scores/labels or the sentinel label -1."""
    data = pd.DataFrame({"label": labels, "score": scores}).dropna()
    data = data[data["label"] != -1]
    return data["label"].to_numpy(dtype=float), data["score"].to_numpy(dtype=float)


def parse_group_indices(value):
    """Parse a JSON list of lists (semantic groups of prompt/sample indices)."""
    if pd.isna(value):
        return []
    try:
        groups = json.loads(value)
    except Exception:
        return []
    parsed = []
    for group in groups:
        indices = []
        for item in group:
            try:
                indices.append(int(item))
            except Exception:
                pass
        if indices:
            parsed.append(indices)
    return parsed


# ------------------------------------------------------------------ loaders

def load_greedy_frames(args, dataname, model):
    frames = []
    for prompt in prompt_names(args.num_answer_prompts):
        path = csv_path(args, dataname, model, "test", "greedy", prompt)
        if os.path.isfile(path):
            frames.append((prompt, read_csv(path)))
        else:
            print(f"Skipping missing greedy file: {path}")
    return frames


def load_sample_frames(args, dataname, model):
    frames = []
    sample_prompt = args.sample_prompt_name or REFERENCE_PROMPT
    for idx in range(args.num_samples):
        path = csv_path(args, dataname, model, "test", "sample", sample_prompt, idx)
        if os.path.isfile(path):
            frames.append((idx, read_csv(path)))
        else:
            print(f"Skipping missing sample file: {path}")
    return frames


def load_supervised_ans0(args, dataname, model):
    """Reference-prompt greedy scores joined with correctness labels."""
    score_path = csv_path(args, dataname, model, "test", "greedy", REFERENCE_PROMPT)
    label_path = csv_path(args, dataname, model, "test", "greedy", REFERENCE_PROMPT, with_gpt_score=True)
    if not os.path.isfile(score_path):
        print(f"Skipping missing ans0 score file: {score_path}")
        return None
    if not os.path.isfile(label_path):
        print(f"Skipping missing ans0 GPT-score file: {label_path}")
        return None

    score_frame = read_csv(score_path)
    label_frame = read_csv(label_path)
    if args.label_col not in label_frame.columns:
        print(f"Skipping supervised metrics for {score_path}: {label_path} lacks {args.label_col}")
        return None

    frame = score_frame.copy()
    frame[args.label_col] = pd.to_numeric(label_frame[args.label_col], errors="coerce").reindex(frame.index)
    return frame


def save_group_metric_columns(path, updates):
    """Write per-question metric values back into a groups CSV."""
    if not updates or not os.path.isfile(path):
        return
    frame = read_csv(path)
    for col, values in updates.items():
        if values is None:
            continue
        frame[col] = pd.Series(values, dtype="float64").reindex(frame.index)
    write_csv(frame, path)
