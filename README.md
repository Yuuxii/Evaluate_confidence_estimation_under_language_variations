# Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations

Code for the paper **[Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations](https://arxiv.org/abs/2601.08064)**
by Yuxi Xia, Dennis Ulmer, Terra Blevins, Yihong Liu, Hinrich Schütze, and Benjamin Roth.
Accepted at the [Third Workshop on Uncertainty-Aware NLP (UncertaiNLP 2026)](https://uncertainlp.github.io/2026/).

## Overview

Evaluations of confidence estimation (CE) for LLMs mostly ask whether confidence aligns with correctness
(ECE, Brier score, AUROC). They ignore *language variability*. The paper proposes three additional properties:

| Property | Metric | What it measures |
|---|---|---|
| **Robustness** | P-RB | Does confidence stay the same when only the prompt wording changes and the (greedy) answer stays semantically equivalent? |
| **Stability** | A-STB | Does confidence stay consistent across sampled answers that mean the same thing? |
| **Sensitivity** | A-SST | Does confidence differ more *between* semantically different answers than *within* a group of equivalent ones? |



## Pipeline

```
 generation ──► white-box scoring ──► judging ──► evaluation
 ce-generate     ce-whitebox          ce-judge     ce-eval
 (vLLM)          (HF transformers)    (GPT-4o)
```


## Repository layout

```
src/ce_variations/
├── config.py                  # settings shared by all stages
├── generation/
│   ├── models.py              # model registry + vLLM loading
│   ├── load_data_prompt.py    # datasets and prompt templates
│   ├── infer.py               # vLLM answers / verbalized confidence
│   └── white_box.py           # HF answers + likelihood, attention, hidden-state scores
├── judging/
│   ├── prompts.py             # correctness and grouping prompts
│   └── judge.py               # GPT-4o correctness judging (Batch API)
└── evaluation/
    ├── config.py              # CE methods, score-column aliases
    ├── data.py                # file naming, CSV loading, group parsing
    ├── metrics.py             # ECE, Brier, AUROC
    ├── variation.py           # P-RB, A-STB, A-SST
    ├── evaluate.py            # per-(dataset, model) evaluation
    ├── report.py              # summary / rank tables, LaTeX, plot
    └── cli.py
```

## Installation

```bash
pip install -e .                 # evaluation only
pip install -e ".[judge]"        # + GPT-4o judging (openai, pyyaml)
pip install -e ".[generation]"   # + generation (vllm, torch, transformers, datasets)
```

## Usage

### 1. Generate responses

```bash
# greedy answers on the training split
python -m ce_variations.generation --model olmo --train
```


### 2. White-box scores

```bash
python -m ce_variations.generation.white_box --model olmo --split test --dataname nq
```


### 3. Judge correctness with GPT-4o

Put your key in `key.yml` (`openai_key: sk-...`, git-ignored) or set `OPENAI_API_KEY`.

```bash
python -m ce_variations.judging --model olmo --split test --prompt_name ans0
```


### 4. Evaluate

```bash
python -m ce_variations.evaluation \
    --models <model_name> [<model_name> ...] \
    --datanames nq sciq triviaqa popqa \
    --response_dir responses \
    --output_dir results
```

## Citation

```bibtex
@misc{xia2026calibration,
  title         = {Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations},
  author        = {Xia, Yuxi and Ulmer, Dennis and Blevins, Terra and Liu, Yihong and Sch{\"u}tze, Hinrich and Roth, Benjamin},
  year          = {2026},
  eprint        = {2601.08064},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.08064}
}
```

## License

See [LICENSE](LICENSE).
