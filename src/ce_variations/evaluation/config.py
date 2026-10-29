"""Evaluation constants: CE methods, score-column aliases, summary columns."""

from ..config import (  # noqa: F401  (re-exported for the evaluation modules)
    DEFAULT_DATASETS,
    DEFAULT_LABEL_COL,
    DEFAULT_RESPONSE_DIR,
    PROMPT_PREFIX,
    REFERENCE_PROMPT,
)

DEFAULT_OUTPUT_DIR = "results"

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

# CSV column(s) holding each method's confidence score (first match wins).
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
# Sampled-answer files store verbalized confidence under a different column.
SAMPLE_METHOD_ALIASES = {
    **METHOD_ALIASES,
    "Verbalized Conf.": ["vc_score"],
}

RANK_LOWER_IS_BETTER = {"brier"}
HIGHER_IS_BETTER = {"auroc", "sensitivity", "robustness", "stability"}

SUMMARY_COLS = [
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
