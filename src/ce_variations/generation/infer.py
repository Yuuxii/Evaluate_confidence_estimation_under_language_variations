"""Generate answers and verbalized confidences with vLLM.

Per dataset and model this fills `{response_dir}/{dataset}_{model}_{split}.csv`:

* `--train`: one greedy answer per question under the `ans0` prompt
  (`response`, `response_prob`).
* test, first run: greedy answers under an answer prompt
  (`{ans}_response`, `{ans}_response_prob`), or with `--sampling`
  `--num_samples` sampled answers (`sampling_{i}_response`, `sampling_{i}_response_prob`).
* test, once sampled answers exist: verbalized confidence for each sample
  (`sampling_{i}_response_vc`).

`response_prob` is the geometric-mean token probability (sequence likelihood).
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd

from ..config import DEFAULT_DATASETS, DEFAULT_RESPONSE_DIR, REFERENCE_PROMPT
from .load_data_prompt import QDataset, SQADataset, answer_type_dict, load_data
from .models import MODEL_SPECS, load_model

MAX_NEW_TOKENS = 50


def build_sampling_params(sampling):
    from vllm import SamplingParams

    if sampling:
        return SamplingParams(temperature=0.7, top_p=0.9, max_tokens=MAX_NEW_TOKENS, logprobs=1)
    return SamplingParams(temperature=0, top_p=1.0, max_tokens=MAX_NEW_TOKENS, top_k=1, n=1, logprobs=1)


def llm_inference(llm, dataloader, sampling=False):
    """Return (responses, sequence probabilities) for every batch of prompts."""
    sampling_params = build_sampling_params(sampling)
    responses, probs = [], []
    for prompts in dataloader:
        for result in llm.generate(prompts, sampling_params):
            output = result.outputs[0]
            responses.append(output.text)
            probs.append(np.exp(output.cumulative_logprob / len(output.token_ids)))
    return responses, probs


def make_dataloader(dataset, batch_size):
    import torch

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)


def response_csv_path(args, dataname):
    split = "train" if args.train else "test"
    return os.path.join(args.response_dir, f"{dataname}_{args.model}_{split}.csv")


def load_or_init_responses(path, data):
    if os.path.isfile(path):
        return pd.read_csv(path, encoding="utf-8", index_col=False)
    return pd.DataFrame({
        "question": [item["question"] for item in data],
        "target": [item["target"] for item in data],
    })


def generate_train(args, llm, tokenizer, data, response):
    """Greedy `ans0` answers on the training split."""
    loader = make_dataloader(QDataset(data, ans_type=REFERENCE_PROMPT, tokenizer=tokenizer), args.bs)
    response["response"], response["response_prob"] = llm_inference(llm, loader)


def generate_test(args, llm, tokenizer, data, response):
    """Greedy answers per prompt variant, or `--num_samples` sampled answers."""
    if args.ans_type == "all":
        for ans_type in answer_type_dict.keys():
            loader = make_dataloader(QDataset(data=data, ans_type=ans_type, tokenizer=tokenizer), args.bs)
            response[f"{ans_type}_response"], response[f"{ans_type}_response_prob"] = llm_inference(llm, loader)
        return

    loader = make_dataloader(QDataset(data=data, ans_type=args.ans_type, tokenizer=tokenizer), args.bs)
    if args.sampling:
        for i in range(args.num_samples):
            response[f"sampling_{i}_response"], response[f"sampling_{i}_response_prob"] = llm_inference(
                llm, loader, sampling=True
            )
    else:
        response[f"{args.ans_type}_response"], response[f"{args.ans_type}_response_prob"] = llm_inference(
            llm, loader
        )


def verbalize_sample_confidence(args, llm, tokenizer, response):
    """Ask the model to verbalize its confidence in each sampled answer."""
    for idx in range(args.num_samples):
        loader = make_dataloader(SQADataset(responses=response, idx=idx, tokenizer=tokenizer), args.bs)
        response[f"sampling_{idx}_response_vc"], _ = llm_inference(llm, loader)


def run_dataset(args, llm, tokenizer, dataname):
    path = response_csv_path(args, dataname)
    train_data, test_data = load_data(dataname)
    data = train_data if args.train else test_data
    response = load_or_init_responses(path, data)

    if args.train:
        if "response" not in response.columns:
            generate_train(args, llm, tokenizer, data, response)
        else:
            print(f"Skipping {path}: training responses already exist.")
    elif "sampling_1_response" not in response.columns:
        generate_test(args, llm, tokenizer, data, response)
    else:
        print("Sampled answers found: eliciting verbalized confidence.")
        verbalize_sample_confidence(args, llm, tokenizer, response)

    response = response.loc[:, ~response.columns.str.contains("^Unnamed")]
    os.makedirs(args.response_dir, exist_ok=True)
    response.to_csv(path, encoding="utf-8", index=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate LLM answers and verbalized confidences.")
    parser.add_argument("--model", default="olmo", choices=sorted(MODEL_SPECS))
    parser.add_argument("--train", action="store_true", help="Generate on the training split (ans0 only).")
    parser.add_argument("--ans_type", default=REFERENCE_PROMPT,
                        help="Answer prompt (e.g. ans0), or 'all' to run every prompt variant.")
    parser.add_argument("--sampling", action="store_true", help="Sample answers instead of greedy decoding.")
    parser.add_argument("--num_samples", default=10, type=int, help="Number of sampled answers per question.")
    parser.add_argument("--bs", default=500, type=int, help="Batch size.")
    parser.add_argument("--num_gpu", default=2, type=int, help="GPUs for tensor parallelism.")
    parser.add_argument("--cache_dir", default=None, help="Model download directory.")
    parser.add_argument("--datanames", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--response_dir", default=DEFAULT_RESPONSE_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("args:", args)

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    llm, tokenizer = load_model(args.model, num_gpu=args.num_gpu, cache_dir=args.cache_dir)
    for dataname in args.datanames:
        run_dataset(args, llm, tokenizer, dataname)


if __name__ == "__main__":
    main()
