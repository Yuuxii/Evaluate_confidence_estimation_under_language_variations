"""Aggregate per-run metrics into summary/rank tables, LaTeX and plots."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import HIGHER_IS_BETTER, METHOD_ORDER, RANK_LOWER_IS_BETTER, SUMMARY_COLS  # noqa: E402


def aggregate_table(results):
    frame = pd.DataFrame(results)
    summary = frame.groupby("method")[SUMMARY_COLS].mean().reset_index()
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    summary["_order"] = summary["method"].map(order).fillna(len(order))
    return summary.sort_values(["_order", "method"]).drop(columns="_order").reset_index(drop=True)


def rank_table(results):
    """Average rank of each method over (dataset, model) pairs and metrics."""
    frame = pd.DataFrame(results)
    rank_cols = [f"{metric}_rank" for metric in sorted(RANK_LOWER_IS_BETTER | HIGHER_IS_BETTER)]
    rank_rows = []
    for _, group in frame.groupby(["dataset", "model"]):
        ranked = group[["method"]].copy()
        for metric in RANK_LOWER_IS_BETTER:
            ranked[f"{metric}_rank"] = group[metric].rank(ascending=True, na_option="bottom")
        for metric in HIGHER_IS_BETTER:
            ranked[f"{metric}_rank"] = group[metric].rank(ascending=False, na_option="bottom")
        ranked["overall_rank"] = ranked[rank_cols].mean(axis=1)
        rank_rows.append(ranked[["method", *rank_cols, "overall_rank"]])

    ranks = pd.concat(rank_rows).groupby("method").mean(numeric_only=True).reset_index()
    return ranks.sort_values("overall_rank").reset_index(drop=True)


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


def save_outputs(results, accuracy_rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw = pd.DataFrame(results)
    summary = aggregate_table(results)
    ranks = rank_table(results)
    accuracy = pd.DataFrame(accuracy_rows)

    raw.to_csv(os.path.join(output_dir, "metrics_raw.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    ranks.to_csv(os.path.join(output_dir, "overall_ranks.csv"), index=False)
    accuracy.to_csv(os.path.join(output_dir, "accuracy.csv"), index=False)

    latex = summary.set_index("method").to_latex(
        float_format="%.3f", caption="Confidence estimation metrics.", label="tab:ce_metrics"
    )
    with open(os.path.join(output_dir, "metrics_summary.tex"), "w", encoding="utf-8") as f:
        f.write(latex)
    accuracy_latex = accuracy.pivot(index="model", columns="dataset", values="accuracy").to_latex(
        float_format="%.3f",
        caption="Greedy ans0 answer accuracy by model and dataset.",
        label="tab:ce_accuracy",
    )
    with open(os.path.join(output_dir, "accuracy.tex"), "w", encoding="utf-8") as f:
        f.write(accuracy_latex)

    plot_ranks(ranks, os.path.join(output_dir, "overall_rank.png"))
    print(latex)
    print(f"Saved results to {output_dir}")
