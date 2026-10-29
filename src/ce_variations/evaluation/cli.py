"""Command-line entry point: `python -m ce_variations` or `ce-eval`."""

import argparse

from .config import (
    DEFAULT_DATASETS,
    DEFAULT_LABEL_COL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESPONSE_DIR,
    METHOD_ORDER,
)
from .evaluate import accuracy_one, evaluate_one
from .report import save_outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate CE methods on calibration, robustness, stability and sensitivity."
    )
    parser.add_argument("--models", nargs="+", required=True, help="Model names as used in the response file names.")
    parser.add_argument("--datanames", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--response_dir", default=DEFAULT_RESPONSE_DIR, help="Directory with response CSVs.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label_col", default=DEFAULT_LABEL_COL, help="Correctness-label column (judge score).")
    parser.add_argument("--sample_prompt_name", default=None, help="Prompt used for sampled answers (default: ans0).")
    parser.add_argument("--num_answer_prompts", default=10, type=int, help="Number of prompt variants (ans0..ansM-1).")
    parser.add_argument("--num_samples", default=10, type=int, help="Number of sampled answers per question.")
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    parser.add_argument(
        "--save_group_scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-question metric columns back into the *_groups.csv files.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = []
    accuracy_rows = []
    for model in args.models:
        for dataname in args.datanames:
            accuracy_row = accuracy_one(args, dataname, model)
            accuracy_rows.append(accuracy_row)
            results.extend(evaluate_one(args, dataname, model, accuracy_row["accuracy"]))
    if not results:
        raise ValueError("No metrics were computed. Check --models, --datanames, and --response_dir.")
    save_outputs(results, accuracy_rows, args.output_dir)


if __name__ == "__main__":
    main()
