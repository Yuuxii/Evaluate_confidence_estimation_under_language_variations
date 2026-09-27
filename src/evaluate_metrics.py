import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import load_data_prompt


DEFAULT_DATASETS = ["nq", "sciq", "triviaqa", "popqa"]
DEFAULT_RESPONSE_DIR = "/mnt/nlp-data/home/users/xiay41cs/responses"
DEFAULT_OUTPUT_DIR = "results/new_csv_metrics"
ANSWER_PROMPTS = getattr(
    load_data_prompt,
    "answer_type_dict",
    {f"ans{i}": load_data_prompt.answer_type_dict for i in range(10)},
)

METHOD_ORDER = [
    "Seq. Likelihood",
    "Boosted Prob.",
    "Platt Scaling",
    "P(True)",
    "Attention Score",
    "Hidden Score",
    "SAPLMA",
    "P(IK)",
    "Verbalized Conf.",
    "Calib1",
]
METHOD_ALIASES = {
    "Seq. Likelihood": ["response_prob"],
    "Boosted Prob.": ["boostedprob_score", "boosted_prob", "boosted_prob_score"],
    "Platt Scaling": ["ps_score"],
    "P(True)": ["p_true_score"],
    "Attention Score": ["norm_attention_score", "attention_score_calibrated"],
    "Hidden Score": ["norm_hidden_score", "hidden_score_calibrated"],
    "SAPLMA": ["saplma_score", "salmpa_score"],
    "P(IK)": ["pik_score"],
    "Verbalized Conf.": ["vc_ans_score"],
    "Calib1": ["infosel_score"],
}
SAMPLE_METHOD_ALIASES = {
    **METHOD_ALIASES,
    "Verbalized Conf.": ["vc_score"],
}
LOWER_IS_BETTER = {"ece", "brier"}
RANK_LOWER_IS_BETTER = {"brier"}
HIGHER_IS_BETTER = {"auroc", "sensitivity", "robustness", "stability"}


def read_csv(path):
    frame = pd.read_csv(path, encoding="utf-8")
    return frame.loc[:, ~frame.columns.str.contains("^Unnamed")]


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8")


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


def score_col(frame, method, sample=False):
    aliases = SAMPLE_METHOD_ALIASES if sample else METHOD_ALIASES
    for col in aliases[method]:
        if col in frame.columns:
            return col
    return None


def numeric_series(frame, method, sample=False):
    col = score_col(frame, method, sample=sample)
    if col is None:
        return None
    values = pd.to_numeric(frame[col], errors="coerce")
    values = values.where((values >= 0) & (values <= 1))
    return values


def valid_xy(labels, scores):
    data = pd.DataFrame({"label": labels, "score": scores}).dropna()
    data = data[data["label"] != -1]
    return data["label"].to_numpy(dtype=float), data["score"].to_numpy(dtype=float)


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


def new_auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique((labels >= 0.5).astype(int))) < 2:
        return np.nan
    return float(roc_auc_score((labels >= 0.5).astype(int), scores))


def brier(labels, scores):
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


def method_col_prefix(method):
    prefix = re.sub(r"[^0-9a-zA-Z]+", "_", str(method).lower()).strip("_")
    return prefix or "method"


def row_score(values, invert_std=False):
    row_std = score_std(values)
    if pd.isna(row_std):
        return np.nan
    return 1 - row_std if invert_std else row_std


def evaluated_count(values):
    if values is None:
        return 0, 0
    return sum(pd.notna(value) for value in values), len(values)


def coverage_text(name, values):
    used, total = evaluated_count(values)
    pct = 100 * used / total if total else np.nan
    return f"{name}: {used}/{total} ({pct:.1f}%)" if total else f"{name}: 0/0 (nan%)"


def diff(list1, list2):
    list1 = [x for x in list1 if pd.notna(x)]
    list2 = [x for x in list2 if pd.notna(x)]
    if len(list1) < 2 or not list2:
        return np.nan
    ingroup = [abs(x - y) for x in list1 for y in list1]
    outgroup = [abs(x - y) for x in list1 for y in list2]
    return float(abs(np.mean(outgroup) - np.mean(ingroup)))


def load_greedy_frames(args, dataname, model):
    frames = []
    prompt_names = list(ANSWER_PROMPTS.keys())[: args.num_answer_prompts]
    for prompt in prompt_names:
        path = csv_path(args, dataname, model, "test", "greedy", prompt)
        if os.path.isfile(path):
            frames.append((prompt, read_csv(path)))
        else:
            print(f"Skipping missing greedy file: {path}")
    return frames


def load_supervised_ans0(args, dataname, model):
    score_path = csv_path(args, dataname, model, "test", "greedy", "ans0")
    label_path = csv_path(args, dataname, model, "test", "greedy", "ans0", with_gpt_score=True)
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


def accuracy_one(args, dataname, model):
    label_path = csv_path(args, dataname, model, "test", "greedy", "ans0", with_gpt_score=True)
    if not os.path.isfile(label_path):
        print(f"Skipping accuracy for {dataname}/{model}: missing {label_path}")
        return {
            "dataset": dataname,
            "model": model,
            "accuracy": np.nan,
            "num_examples": 0,
        }
    frame = read_csv(label_path)
    if args.label_col not in frame.columns:
        print(f"Skipping accuracy for {dataname}/{model}: {label_path} lacks {args.label_col}")
        return {
            "dataset": dataname,
            "model": model,
            "accuracy": np.nan,
            "num_examples": 0,
        }
    labels = pd.to_numeric(frame[args.label_col], errors="coerce")
    labels = labels[(labels.notna()) & (labels != -1)]
    return {
        "dataset": dataname,
        "model": model,
        "accuracy": float(labels.mean()) if len(labels) else np.nan,
        "num_examples": int(len(labels)),
    }


def load_sample_frames(args, dataname, model):
    frames = []
    sample_prompt = args.sample_prompt_name or next(iter(ANSWER_PROMPTS.keys()))
    for idx in range(args.num_samples):
        path = csv_path(args, dataname, model, "test", "sample", sample_prompt, idx)
        if os.path.isfile(path):
            frames.append((idx, read_csv(path)))
        else:
            print(f"Skipping missing sample file: {path}")
    return frames


def metric_ece_brier_auroc(ans0, method, label_col):
    if ans0 is None:
        return np.nan, np.nan, np.nan
    if label_col not in ans0.columns:
        return np.nan, np.nan, np.nan
    scores = numeric_series(ans0, method)
    if scores is None:
        return np.nan, np.nan, np.nan
    labels, scores = valid_xy(pd.to_numeric(ans0[label_col], errors="coerce"), scores)
    return compute_ece(labels, scores), brier(labels, scores), new_auroc(labels, scores)


def group_containing_ans0(groups):
    for group in groups:
        if 0 in group:
            return group
    return []


def prompt_robustness_values(args, dataname, model, greedy_frames, method):
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

    valid = [value for value in values if pd.notna(value)]
    valid_surviving = [value for value in surviving_prompt_fractions if pd.notna(value)]
    valid_all_agree = [value for value in all_agree_values if pd.notna(value)]
    valid_partial_agree = [value for value in partial_agree_values if pd.notna(value)]
    stats = {
        "surviving_prompt_fraction": float(np.nanmean(valid_surviving)) if valid_surviving else np.nan,
        "all_agree_fraction": float(np.nanmean([value == 1.0 for value in valid_surviving])) if valid_surviving else np.nan,
        "partial_agree_fraction": float(np.nanmean([(value < 1.0 and value > 0.0) for value in valid_surviving]))
        if valid_surviving
        else np.nan,
        "robustness_all_agree": float(np.nanmean(valid_all_agree)) if valid_all_agree else np.nan,
        "robustness_partial_agree": float(np.nanmean(valid_partial_agree)) if valid_partial_agree else np.nan,
        "surviving_prompt_fraction_values": surviving_prompt_fractions,
        "robustness_all_agree_values": all_agree_values,
        "robustness_partial_agree_values": partial_agree_values,
    }
    return float(np.nanmean(valid)) if valid else np.nan, values, stats


def parse_prompt_index(prompt_name):
    text = str(prompt_name)
    if text.startswith("ans"):
        try:
            return int(text[3:])
        except ValueError:
            return None
    return None


def parse_group_indices(value):
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
    candidates = []
    for group in groups:
        valid_group = [i for i in group if i in values]
        if len(valid_group) >= 2 and has_distinct_answer_text(valid_group, answers):
            candidates.append(valid_group)
    if not candidates:
        return []
    return max(candidates, key=len)


def sample_stability_sensitivity_values(args, dataname, model, sample_frames, method):
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
            stability_scores = [values[i] for i in stability_group if i in values]
            stability_values[row_idx] = row_score(stability_scores, invert_std=True)
        if largest_index != smallest_index and len(largest) >= 2 and smallest:
            sensitivity_values[row_idx] = diff(largest, smallest)

    valid_stability = [value for value in stability_values if pd.notna(value)]
    valid_sensitivity = [value for value in sensitivity_values if pd.notna(value)]

    return (
        float(np.nanmean(valid_stability)) if valid_stability else np.nan,
        float(np.nanmean(valid_sensitivity)) if valid_sensitivity else np.nan,
        stability_values,
        sensitivity_values,
    )


def save_group_metric_columns(path, updates):
    if not updates or not os.path.isfile(path):
        return
    frame = read_csv(path)
    for col, values in updates.items():
        if values is None:
            continue
        series = pd.Series(values, dtype="float64")
        frame[col] = series.reindex(frame.index)
    write_csv(frame, path)


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
        ece, brier_score, auroc = metric_ece_brier_auroc(ans0, method, args.label_col)
        robustness, robustness_values, robustness_stats = prompt_robustness_values(
            args, dataname, model, greedy_frames, method
        )
        stability, sensitivity, stability_values, sensitivity_values = sample_stability_sensitivity_values(
            args, dataname, model, sample_frames, method
        )
        prefix = method_col_prefix(method)
        greedy_updates[f"{prefix}_robustness"] = robustness_values
        greedy_updates[f"{prefix}_surviving_prompt_fraction"] = robustness_stats.get("surviving_prompt_fraction_values")
        greedy_updates[f"{prefix}_robustness_all_agree"] = robustness_stats.get("robustness_all_agree_values")
        greedy_updates[f"{prefix}_robustness_partial_agree"] = robustness_stats.get("robustness_partial_agree_values")
        sample_updates[f"{prefix}_stability"] = stability_values
        sample_updates[f"{prefix}_sensitivity"] = sensitivity_values
        print(
            f"Coverage {dataname}/{model}/{method}: "
            f"{coverage_text('P-RB', robustness_values)}, "
            f"{coverage_text('A-STB', stability_values)}, "
            f"{coverage_text('A-SST', sensitivity_values)}, "
            f"|I|/M={robustness_stats.get('surviving_prompt_fraction', np.nan):.3f}, "
            f"P-RB_all={robustness_stats.get('robustness_all_agree', np.nan):.3f}, "
            f"P-RB_partial={robustness_stats.get('robustness_partial_agree', np.nan):.3f}"
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
                "surviving_prompt_fraction": robustness_stats.get("surviving_prompt_fraction", np.nan),
                "all_agree_fraction": robustness_stats.get("all_agree_fraction", np.nan),
                "partial_agree_fraction": robustness_stats.get("partial_agree_fraction", np.nan),
                "robustness_all_agree": robustness_stats.get("robustness_all_agree", np.nan),
                "robustness_partial_agree": robustness_stats.get("robustness_partial_agree", np.nan),
                "stability": stability,
                "sensitivity": sensitivity,
            }
        )
    if args.save_group_scores:
        save_group_metric_columns(greedy_groups_path(args, dataname, model), greedy_updates)
        save_group_metric_columns(groups_path(args, dataname, model), sample_updates)
    return rows


def aggregate_table(results):
    frame = pd.DataFrame(results)
    summary_cols = [
        "accuracy",
        "ece",
        "brier",
        "auroc",
        "robustness",
        "surviving_prompt_fraction",
        "all_agree_fraction",
        "partial_agree_fraction",
        "robustness_all_agree",
        "robustness_partial_agree",
        "stability",
        "sensitivity",
    ]
    summary = frame.groupby("method")[summary_cols].mean().reset_index()
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    summary["_order"] = summary["method"].map(order).fillna(len(order))
    return summary.sort_values(["_order", "method"]).drop(columns="_order").reset_index(drop=True)


def rank_table(results):
    frame = pd.DataFrame(results)
    rank_rows = []
    for (_, _), group in frame.groupby(["dataset", "model"]):
        ranked = group[["method"]].copy()
        for metric in RANK_LOWER_IS_BETTER:
            ranked[f"{metric}_rank"] = group[metric].rank(ascending=True, na_option="bottom")
        for metric in HIGHER_IS_BETTER:
            ranked[f"{metric}_rank"] = group[metric].rank(ascending=False, na_option="bottom")
        rank_cols = [f"{metric}_rank" for metric in sorted(RANK_LOWER_IS_BETTER | HIGHER_IS_BETTER)]
        ranked["overall_rank"] = ranked[rank_cols].mean(axis=1)
        rank_rows.append(ranked[["method", *rank_cols, "overall_rank"]])

    ranks = pd.concat(rank_rows).groupby("method").mean(numeric_only=True).reset_index()
    return ranks.sort_values("overall_rank").reset_index(drop=True)


def save_outputs(results, accuracy_rows, args):
    os.makedirs(args.output_dir, exist_ok=True)
    raw = pd.DataFrame(results)
    summary = aggregate_table(results)
    ranks = rank_table(results)
    accuracy = pd.DataFrame(accuracy_rows)

    raw.to_csv(os.path.join(args.output_dir, "metrics_raw.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)
    ranks.to_csv(os.path.join(args.output_dir, "overall_ranks.csv"), index=False)
    accuracy.to_csv(os.path.join(args.output_dir, "accuracy.csv"), index=False)

    latex = summary.set_index("method").to_latex(float_format="%.3f", caption="New CSV uncertainty metrics.", label="tab:new_csv_metrics")
    with open(os.path.join(args.output_dir, "metrics_summary.tex"), "w", encoding="utf-8") as f:
        f.write(latex)
    accuracy_latex = accuracy.pivot(index="model", columns="dataset", values="accuracy").to_latex(
        float_format="%.3f",
        caption="Greedy ans0 answer accuracy by model and dataset.",
        label="tab:new_csv_accuracy",
    )
    with open(os.path.join(args.output_dir, "accuracy.tex"), "w", encoding="utf-8") as f:
        f.write(accuracy_latex)

    plot_ranks(ranks, os.path.join(args.output_dir, "overall_rank.png"))
    print(latex)
    print(f"Saved results to {args.output_dir}")


def plot_ranks(ranks, path):
    ranks = ranks.sort_values("overall_rank", ascending=True)
    plt.figure(figsize=(8, max(4, 0.35 * len(ranks))))
    plt.barh(ranks["method"], ranks["overall_rank"], color="#4C78A8")
    plt.gca().invert_yaxis()
    plt.xlabel("Average rank across datasets, models, and metrics (lower is better)")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def default_models():
    try:
        from white_llm_inference import MODEL_NAME_MAP

        return list(MODEL_NAME_MAP.keys())
    except Exception:
        return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datanames", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--response_dir", default=DEFAULT_RESPONSE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label_col", default="gpt_score")
    parser.add_argument("--sample_prompt_name", default=None)
    parser.add_argument("--num_answer_prompts", default=10, type=int)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    parser.add_argument("--save_group_scores", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.models is None:
        args.models = default_models()
        if not args.models:
            raise ValueError("Please provide --models.")
    return args


def main():
    args = parse_args()
    results = []
    accuracy_rows = []
    for model in args.models:
        for dataname in args.datanames:
            accuracy_row = accuracy_one(args, dataname, model)
            accuracy_rows.append(accuracy_row)
            results.extend(evaluate_one(args, dataname, model, accuracy_row["accuracy"]))
    if not results:
        raise ValueError("No metrics were computed. Check --models, --datanames, and response_dir.")
    save_outputs(results, accuracy_rows, args)


if __name__ == "__main__":
    main()
