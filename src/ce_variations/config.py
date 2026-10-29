"""Settings shared by the generation, judging and evaluation stages."""

DEFAULT_DATASETS = ["nq", "sciq", "triviaqa", "popqa"]
DEFAULT_RESPONSE_DIR = "responses"
DEFAULT_LABEL_COL = "gpt_score"

# Prompt variants are named ans0, ans1, ...; ans0 is the reference prompt.
PROMPT_PREFIX = "ans"
REFERENCE_PROMPT = f"{PROMPT_PREFIX}0"
