"""GPT-4o judge prompts: answer correctness and semantic-equivalence grouping."""

_INSTRUCTIONS = (
    "Also, treat answers as equivalent even if they include expressions of confidence "
    "or uncertainty (e.g., 'I'm pretty sure it's X' vs. 'X'). Focus on the core meaning "
    "of the answers and ignore differences in:\n"
    "- wording or phrasing\n"
    "- length\n"
    "- fillers\n"
    "- strengtheners or weakeners\n\n"
    "Examples:\n"
    "- Answer: 'Mount Everest' | Reference: 'Everest' → YES\n"
    "- Answer: 'Albert Einstein' | Reference: 'Einstein' → YES\n"
    "- Answer: 'Paris' | Reference: 'The capital of France' → YES\n"
    "- Answer: 'Pacific Ocean' | Reference: 'Atlantic Ocean' → NO\n"
    "- Answer: 'Jupiter' | Reference: 'I think it's Jupiter' → YES\n"
    "- Answer: 'Mars' | Reference: 'Maybe Venus' → NO\n\n"
)


def build_correctness_prompt(question, answer, target):
    """Prompt asking whether `answer` is semantically equivalent to `target` (reply 1 / 0)."""
    return (
        "You are evaluating whether an answer is semantically equivalent to the reference answer. "
        "Your goal is to judge meaning, not exact wording. "
        "Two answers should be rated the same if they are semantically equivalent, "
        "even if they use different phrasing, synonyms, or slightly different names that refer to the same entity.\n\n"
        + _INSTRUCTIONS
        + "Answer the equivalent only with:\n"
        "1, if they are semantically equivalent\n"
        "0, if they are not.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Reference answer: {target}\n"
        "Equivalent: "
    )


def build_grouping_prompt(question, answers):
    """Prompt asking the judge to group 10 sampled answers by semantic equivalence."""
    return (
        "You are grouping 10 answers by their semantic equivalence. "
        "Your goal is to judge meaning, not exact wording. Two answers should "
        "belong to one group if they are semantically equivalent, even if they use "
        "different phrasing, synonyms, or slightly different names that refer to the same entity.\n\n"
        + _INSTRUCTIONS
        + "Answer in a dict format only, answers in the same group fall into one dict, for example:\n"
        "{1: answer1, 2: ...}\n"
        "{4: answer4, 5: ...}\n\n"
        f"Question: {question}\n"
        f"Answers: {answers}"
    )


def create_answer_list(responses, num_samples=10):
    """Map sample index -> column of sampled answers in a response DataFrame."""
    return {i: responses[f"sampling_{i}_response"] for i in range(num_samples)}
