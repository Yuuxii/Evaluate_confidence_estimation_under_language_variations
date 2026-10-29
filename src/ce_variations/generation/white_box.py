import argparse
import gc
import json
import math
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import DEFAULT_DATASETS, DEFAULT_RESPONSE_DIR
from . import load_data_prompt
from .common_utils import get_attn_eig_prod, get_svd_eval, perplexity
from .load_data_prompt import load_data
from .ministral_model_utils import (
    apply_user_chat_template,
    decode_token_ids,
    encode_prompts,
    is_ministral_model,
    load_ministral_model,
    plain_token_ids,
    tokenizer_special_id,
)
from .models import MODEL_NAME_MAP


MANIFEST_COLUMNS = [
    "row_id",
    "idx",
    "experiment",
    "prompt_name",
    "sample_idx",
    "question",
    "target",
    "response",
    "response_prob",
    "generated_token_count",
    "boostedprob_score",
    "perplexity",
    "attention_score",
    "hidden_score",
    "attention_layer_idx",
    "prompt_length",
    "seq_len",
    "feature_path",
]


BOOSTEDPROB_EPSILON = 0.005
BOOSTEDPROB_P_JUMP = 0.3
BOOSTEDPROB_INTERNAL_K = 1024


def prompt_text(prompt_spec):
    if callable(prompt_spec):
        return prompt_spec()
    return str(prompt_spec)


def answer_prompts():
    return [(name, prompt_text(prompt_spec)) for name, prompt_spec in load_data_prompt.answer_type_dict.items()]


class QuestionPromptDataset(Dataset):
    def __init__(self, data, tokenizer, prompt_name, prompt, experiment, sample_idx=None, indices=None):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_name = prompt_name
        self.prompt = prompt
        self.experiment = experiment
        self.sample_idx = sample_idx
        self.indices = list(range(len(data))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        question = self.data[data_idx]["question"].strip()
        content = self.prompt + question
        prompt = apply_user_chat_template(self.tokenizer, content)
        return {
            "idx": data_idx,
            "experiment": self.experiment,
            "prompt_name": self.prompt_name,
            "sample_idx": self.sample_idx,
            "question": question,
            "target": self.data[data_idx]["target"],
            "prompt": prompt,
            "question_token_ids": plain_token_ids(self.tokenizer, question),
            "content_token_ids": plain_token_ids(self.tokenizer, content),
        }


def collate_examples(batch):
    return {
        "idx": [item["idx"] for item in batch],
        "experiment": [item["experiment"] for item in batch],
        "prompt_name": [item["prompt_name"] for item in batch],
        "sample_idx": [item["sample_idx"] for item in batch],
        "question": [item["question"] for item in batch],
        "target": [item["target"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "question_token_ids": [item["question_token_ids"] for item in batch],
        "content_token_ids": [item["content_token_ids"] for item in batch],
    }


def get_model_name(model_key):
    return MODEL_NAME_MAP.get(model_key, model_key)


def load_whitebox_model(args):
    model_name = get_model_name(args.model)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if is_ministral_model(model_name):
        return load_ministral_model(model_name, args, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "cache_dir": args.cache_dir,
        "device_map": args.device_map,
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, tokenizer


def selected_attention_layer(model, layer_ratio):
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(model.config, "n_layer", None)
    if num_layers is None and getattr(model.config, "text_config", None) is not None:
        text_config = model.config.text_config
        num_layers = getattr(text_config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(text_config, "n_layer", None)
    if num_layers is None:
        raise ValueError("Could not infer the number of transformer layers from model.config.")
    return min(num_layers - 1, max(0, int(math.floor((num_layers - 1) * layer_ratio))))


def make_safe_tensor(tensor, dtype):
    if dtype == "float32":
        return tensor.detach().float().cpu()
    return tensor.detach().to(torch.float16).cpu()


def dominant_cut_points(sorted_probs):
    diff = sorted_probs[..., :-1] - sorted_probs[..., 1:]
    mask = (diff > BOOSTEDPROB_P_JUMP * sorted_probs[..., :-1]) & (diff > BOOSTEDPROB_EPSILON)
    padded_mask = torch.zeros_like(sorted_probs, dtype=torch.bool)
    padded_mask[..., :-1] = mask
    mask = padded_mask
    cut_points = mask.shape[-1] - 1 - torch.argmax(torch.flip(mask, dims=[-1]).int(), dim=-1)
    cut_points = cut_points.masked_fill(~mask.any(dim=-1), -1)
    return cut_points


def boostedprob_token_scores(log_probs, target_ids):
    internal_k = min(BOOSTEDPROB_INTERNAL_K, log_probs.shape[-1])
    probs = log_probs.exp()
    sorted_probs, sorted_indices = torch.topk(probs, k=internal_k, dim=-1, largest=True, sorted=True)
    cut_points = dominant_cut_points(sorted_probs)
    ranks = torch.arange(internal_k, device=log_probs.device).view(1, 1, -1)
    dominant_mask = (ranks <= cut_points.unsqueeze(-1)) & (cut_points.unsqueeze(-1) != -1)

    target_in_topk = sorted_indices == target_ids.unsqueeze(-1)
    is_dominant = (target_in_topk & dominant_mask).any(dim=-1)
    dominant_cluster_mass = (sorted_probs * dominant_mask).sum(dim=-1)
    selected_prob = probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return torch.where(is_dominant, dominant_cluster_mass, selected_prob)


def first_stop_position(token_ids, stop_token_ids):
    for pos, token_id in enumerate(token_ids):
        if token_id in stop_token_ids:
            return pos
    return len(token_ids)


def last_generated_token_position(prompt_len, generated_token_count):
    if generated_token_count > 0:
        return prompt_len + generated_token_count - 1
    return max(prompt_len - 1, 0)


def find_last_subsequence(sequence, subsequence, end=None):
    if not subsequence:
        return None
    end = len(sequence) if end is None else end
    for start in range(end - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return None


def question_last_token_position(input_ids, prompt_len, question_token_ids, content_token_ids):
    token_ids = input_ids.squeeze(0).detach().cpu().tolist()
    start = find_last_subsequence(token_ids, question_token_ids, end=prompt_len)
    if start is not None:
        return start + len(question_token_ids) - 1

    start = find_last_subsequence(token_ids, content_token_ids, end=prompt_len)
    if start is not None:
        return start + len(content_token_ids) - 1

    return max(prompt_len - 1, 0)


def model_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def output_value(outputs, name):
    value = getattr(outputs, name, None)
    if value is not None:
        return value
    for nested_name in ("language_model_outputs", "text_model_output", "text_outputs"):
        nested = getattr(outputs, nested_name, None)
        if nested is not None:
            value = getattr(nested, name, None)
            if value is not None:
                return value
    raise AttributeError(f"Model output does not contain {name}.")


def generated_scores_from_outputs(outputs, input_ids, prompt_len, stop_token_ids):
    raw_generated_token_ids = input_ids[:, prompt_len:]
    stop_position = first_stop_position(raw_generated_token_ids[0].detach().cpu().tolist(), stop_token_ids)
    generated_token_ids = raw_generated_token_ids[:, :stop_position]
    generated_token_count = generated_token_ids.shape[1]
    if generated_token_count == 0:
        return math.nan, math.nan, 0

    logits = output_value(outputs, "logits")
    shifted_logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shifted_logits.float(), dim=-1)
    token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

    # token_log_probs[:, t - 1] is log p(input_ids[:, t] | input_ids[:, :t]).
    generated_prediction_start = max(prompt_len - 1, 0)
    gen_logprobs = token_log_probs[
        0,
        generated_prediction_start : generated_prediction_start + generated_token_count,
    ]
    if gen_logprobs.numel() == 0:
        return math.nan, math.nan, 0

    gen_log_probs = log_probs[
        :,
        generated_prediction_start : generated_prediction_start + generated_token_count,
        :,
    ]
    gen_target_ids = labels[
        :,
        generated_prediction_start : generated_prediction_start + generated_token_count,
    ]
    boosted_tokens = boostedprob_token_scores(gen_log_probs, gen_target_ids)
    boostedprob_score = float((boosted_tokens.mean()).detach().cpu())
    response_prob = float(torch.exp(gen_logprobs.mean()).detach().cpu())
    return response_prob, boostedprob_score, generated_token_count


def common_utils_scores(outputs, input_ids, layer_idx, token_span):
    if token_span[1] <= token_span[0]:
        return {"perplexity": math.nan, "attention_score": math.nan, "hidden_score": math.nan}

    hidden_states = output_value(outputs, "hidden_states")
    attentions = output_value(outputs, "attentions")
    logits = output_value(outputs, "logits")
    hidden_acts = [tuple(layer.squeeze(0).float().cpu() for layer in hidden_states)]
    attns = [tuple(layer.squeeze(0).float().cpu() for layer in attentions)]
    logits = [logits.squeeze(0).float().cpu()]
    tok_ins = [input_ids.detach().cpu()]
    tok_lens = [token_span]

    return {
        "perplexity": float(perplexity(logits, tok_ins, tok_lens)[0]),
        "attention_score": float(get_attn_eig_prod(attns, layer_num=layer_idx, tok_lens=tok_lens)[0]),
        "hidden_score": float(get_svd_eval(hidden_acts, layer_num=layer_idx, tok_lens=tok_lens)[0]),
    }


def save_batch_activations(
    model,
    batch,
    generated_ids,
    prompt_lengths,
    input_width,
    responses,
    feature_dir,
    dataname,
    layer_idx,
    tensor_dtype,
    stop_token_ids,
):
    os.makedirs(feature_dir, exist_ok=True)
    rows = []

    for local_idx, dataset_idx in enumerate(batch["idx"]):
        prompt_len = prompt_lengths[local_idx]
        prompt_start = input_width - prompt_len
        input_ids = generated_ids[local_idx : local_idx + 1, prompt_start:]
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        response_prob, boostedprob_score, generated_token_count = generated_scores_from_outputs(
            outputs,
            input_ids,
            prompt_len,
            stop_token_ids,
        )
        last_token_pos = last_generated_token_position(prompt_len, generated_token_count)
        question_token_pos = question_last_token_position(
            input_ids,
            prompt_len,
            batch["question_token_ids"][local_idx],
            batch["content_token_ids"][local_idx],
        )
        scores = common_utils_scores(
            outputs,
            input_ids,
            layer_idx,
            [prompt_len, prompt_len + generated_token_count],
        )
        experiment = batch["experiment"][local_idx]
        prompt_name = batch["prompt_name"][local_idx]
        sample_idx = batch["sample_idx"][local_idx]
        hidden_states = output_value(outputs, "hidden_states")
        suffix = f"{experiment}_{prompt_name}"
        if sample_idx is not None:
            suffix += f"_{sample_idx}"
        feature_path = os.path.join(feature_dir, suffix, f"{dataname}_{dataset_idx:06d}.pt")
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        feature_obj = {
            "last_token_last_layer_hidden_state": make_safe_tensor(
                hidden_states[-1][0, last_token_pos],
                tensor_dtype,
            ),
            "question_last_token_last_layer_hidden_state": make_safe_tensor(
                hidden_states[-1][0, question_token_pos],
                tensor_dtype,
            ),
        }
        torch.save(feature_obj, feature_path)

        rows.append(
            {
                "idx": dataset_idx,
                "experiment": experiment,
                "prompt_name": prompt_name,
                "sample_idx": sample_idx,
                "question": batch["question"][local_idx],
                "target": batch["target"][local_idx],
                "response": responses[local_idx],
                "response_prob": response_prob,
                "generated_token_count": generated_token_count,
                "boostedprob_score": boostedprob_score,
                "perplexity": scores["perplexity"],
                "attention_score": scores["attention_score"],
                "hidden_score": scores["hidden_score"],
                "attention_layer_idx": layer_idx,
                "prompt_length": prompt_len,
                "seq_len": input_ids.shape[1],
                "feature_path": feature_path,
            }
        )

        del outputs

    return rows


def white_llm_inference(model, tokenizer, dataloader, args, dataname, do_sample, temperature):
    pad_token_id = tokenizer_special_id(tokenizer, "pad_token_id")
    eos_token_id = tokenizer_special_id(tokenizer, "eos_token_id")
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "return_dict_in_generate": False,
        "do_sample": do_sample,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": args.top_p,
            }
        )

    feature_dir = os.path.join(args.feature_dir, args.model, dataname, args.split)
    layer_idx = selected_attention_layer(model, args.layer_ratio)
    stop_token_ids = {eos_token_id, pad_token_id}
    stop_token_ids = {token_id for token_id in stop_token_ids if token_id is not None}
    all_rows = []

    for step, batch in enumerate(dataloader):
        enc = encode_prompts(tokenizer, batch["prompt"], args, model_input_device(model))
        prompt_lengths = enc["attention_mask"].sum(dim=1).detach().cpu().tolist()
        input_width = enc["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = model.generate(**enc, **generation_kwargs)

        responses = []
        for row_idx in range(len(prompt_lengths)):
            generated_part = generated_ids[row_idx, input_width:]
            responses.append(decode_token_ids(tokenizer, generated_part).strip())

        rows = save_batch_activations(
            model=model,
            batch=batch,
            generated_ids=generated_ids,
            prompt_lengths=prompt_lengths,
            input_width=input_width,
            responses=responses,
            feature_dir=feature_dir,
            dataname=dataname,
            layer_idx=layer_idx,
            tensor_dtype=args.tensor_dtype,
            stop_token_ids=stop_token_ids,
        )
        for row in rows:
            row["row_id"] = len(all_rows)
            all_rows.append(row)

        del enc, generated_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_rows


def read_existing_manifest(path):
    if not os.path.isfile(path):
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    frame = pd.read_csv(path, encoding="utf-8")
    return frame.loc[:, ~frame.columns.str.contains("^Unnamed")]


def completed_indices(frame):
    if frame.empty or "idx" not in frame.columns:
        return set()
    complete = frame.dropna(subset=["response", "feature_path"])
    complete = complete[complete["feature_path"].map(lambda path: isinstance(path, str) and os.path.isfile(path))]
    return set(complete["idx"].astype(int).tolist())


def reindex_rows(frame):
    frame = frame.sort_values("idx").drop_duplicates(subset=["idx"], keep="last").reset_index(drop=True)
    frame["row_id"] = range(len(frame))
    return frame


def build_dataloader(data, tokenizer, prompt_name, prompt, experiment, sample_idx, args, indices=None):
    dataset = QuestionPromptDataset(
        data=data,
        tokenizer=tokenizer,
        prompt_name=prompt_name,
        prompt=prompt,
        experiment=experiment,
        sample_idx=sample_idx,
        indices=indices,
    )
    return DataLoader(
        dataset,
        batch_size=args.bs,
        drop_last=False,
        collate_fn=collate_examples,
    )


def run_experiment(model, tokenizer, data, dataname, args, prompt_name, prompt, experiment, sample_idx, do_sample, temperature):
    manifest_path = os.path.join(args.response_dir, f"{dataname}_{args.model}_{args.split}_{experiment}_{prompt_name}.csv")
    if sample_idx is not None:
        manifest_path = os.path.join(
            args.response_dir,
            f"{dataname}_{args.model}_{args.split}_{experiment}_{prompt_name}_{sample_idx}.csv",
        )
    existing = read_existing_manifest(manifest_path)
    done = completed_indices(existing)
    remaining = [idx for idx in range(len(data)) if idx not in done]
    if not remaining:
        print(f"Skipping complete experiment: {manifest_path}")
        return reindex_rows(existing).to_dict("records")


    dataloader = build_dataloader(data, tokenizer, prompt_name, prompt, experiment, sample_idx, args, indices=remaining)
    rows = white_llm_inference(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        args=args,
        dataname=dataname,
        do_sample=do_sample,
        temperature=temperature,
    )
    # Create new DataFrame
    new_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    # Only concatenate if the new dataframe has data
    if not new_df.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = existing
    combined = reindex_rows(combined)
    combined.to_csv(manifest_path, encoding="utf-8", index=False)
    print(f"Saved {len(combined)} rows to {manifest_path}")
    return combined.to_dict("records")


def normalize_combined_rows(rows):
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if frame.empty:
        return frame
    sort_cols = ["experiment", "prompt_name", "sample_idx", "idx"]
    frame = frame.sort_values(sort_cols, na_position="first").reset_index(drop=True)
    frame["row_id"] = range(len(frame))
    return frame


def save_run_config(args, output_dir, layer_idx):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({**vars(args), "attention_layer_idx": layer_idx}, f, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate answers with a white-box model and record likelihood, attention and hidden-state scores."
    )
    parser.add_argument("--model", default="olmo", help="Model key from generation/models.py or a Hugging Face model id.")
    parser.add_argument("--dataname", default=None, help="Run one dataset. Defaults to nq, sciq, triviaqa, popqa.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--bs", default=500, type=int)
    parser.add_argument("--max_new_tokens", default=50, type=int)
    parser.add_argument("--max_prompt_length", default=2048, type=int)
    parser.add_argument("--sample_temperature", default=0.7, type=float)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--layer_ratio", default=0.7, type=float, help="Attention layer ratio; 0.7 means about 70% depth.")
    parser.add_argument("--feature_dir", default="model_states")
    parser.add_argument("--response_dir", default=DEFAULT_RESPONSE_DIR)
    parser.add_argument("--tensor_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--cache_dir", default=None, help="Hugging Face cache directory.")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("args:", args)

    os.makedirs(args.response_dir, exist_ok=True)
    model, tokenizer = load_whitebox_model(args)
    layer_idx = selected_attention_layer(model, args.layer_ratio)
    print(f"Computing layer {layer_idx} scores and saving last-layer probe hidden states.")

    datasets_to_run = [args.dataname] if args.dataname is not None else DEFAULT_DATASETS
    all_answer_prompts = answer_prompts()
    answer_prompt_items = all_answer_prompts[:1] if args.split == "train" else all_answer_prompts

    for dataname in datasets_to_run:
        train_data, test_data = load_data(dataname)
        data = train_data if args.split == "train" else test_data
        dataset_rows = []

        for prompt_name, prompt in answer_prompt_items:
            rows = run_experiment(
                model,
                tokenizer,
                data,
                dataname,
                args,
                prompt_name,
                prompt,
                experiment="greedy",
                sample_idx=None,
                do_sample=False,
                temperature=0.0,
            )
            dataset_rows.extend(rows)

        if args.split == "test":
            base_prompt_name, base_prompt = answer_prompt_items[0]
            for sample_idx in range(args.num_samples):
                rows = run_experiment(
                    model,
                    tokenizer,
                    data,
                    dataname,
                    args,
                    base_prompt_name,
                    base_prompt,
                    experiment="sample",
                    sample_idx=sample_idx,
                    do_sample=True,
                    temperature=args.sample_temperature,
                )
                dataset_rows.extend(rows)

        combined_path = os.path.join(args.response_dir, f"{dataname}_{args.model}_{args.split}_white_manifest.csv")
        normalize_combined_rows(dataset_rows).to_csv(combined_path, encoding="utf-8", index=False)
        save_run_config(args, os.path.join(args.feature_dir, args.model, dataname, args.split), layer_idx)
        print(f"Saved combined manifest to {combined_path}")


if __name__ == "__main__":
    main()
