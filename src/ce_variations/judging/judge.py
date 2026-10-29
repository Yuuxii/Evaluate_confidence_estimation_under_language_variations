"""Stage 2: judge answer correctness with GPT-4o through the OpenAI Batch API.

Reads `{dataset}_{model}_{split}_{experiment}_{prompt}.csv` (columns question, target,
response) and writes `..._with_gpt_score.csv` with the judge verdict in `gpt_score`.
Requests are chunked into JSONL batch files; reruns resume and only judge missing rows.
"""

import argparse
import json
import math
import os
import time

import pandas as pd

from ..config import DEFAULT_DATASETS, DEFAULT_LABEL_COL, DEFAULT_RESPONSE_DIR, REFERENCE_PROMPT
from ..generation.models import MODEL_NAME_MAP
from .prompts import build_correctness_prompt


def clean_frame(frame):
    return frame.loc[:, ~frame.columns.str.contains("^Unnamed")]


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def default_csv_path(args, dataname, model, split):
    return os.path.join(
        args.response_dir,
        f"{dataname}_{model}_{split}_{args.experiment}_{args.prompt_name}.csv",
    )


def output_csv_path(args, input_path):
    if args.output_csv is not None:
        return args.output_csv
    if args.overwrite:
        return input_path
    base, ext = os.path.splitext(os.path.basename(input_path))
    output_dir = args.response_dir if args.output_dir is None else args.output_dir
    return os.path.join(output_dir, f"{base}_with_gpt_score{ext}")


def load_response_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return clean_frame(pd.read_csv(path, encoding="utf-8", index_col=False))


def rows_to_judge(frame, args):
    if args.response_col not in frame.columns:
        raise ValueError(f"Missing response column: {args.response_col}")
    if args.target_col not in frame.columns:
        raise ValueError(f"Missing target column: {args.target_col}")
    if args.question_col not in frame.columns:
        raise ValueError(f"Missing question column: {args.question_col}")

    if args.output_col not in frame.columns:
        return list(range(len(frame)))
    if args.force:
        return list(range(len(frame)))
    missing = frame[args.output_col].isna() | (frame[args.output_col].astype(str).str.strip() == "")
    return frame.index[missing].tolist()


def write_batch_inputs(frame, row_indices, args, file_prefix):
    os.makedirs(args.work_dir, exist_ok=True)
    paths = []
    for chunk_idx in range(math.ceil(len(row_indices) / args.chunk_size)):
        chunk = row_indices[chunk_idx * args.chunk_size : (chunk_idx + 1) * args.chunk_size]
        path = os.path.join(args.work_dir, f"{file_prefix}_chunk_{chunk_idx}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row_idx in chunk:
                row = frame.loc[row_idx]
                request = {
                    "custom_id": str(row_idx),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.judge_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": build_correctness_prompt(
                                    clean_text(row[args.question_col]),
                                    clean_text(row[args.response_col]),
                                    clean_text(row[args.target_col]),
                                ),
                            }
                        ],
                        "max_tokens": args.max_tokens,
                        "logprobs": False,
                        "temperature": 0,
                    },
                }
                f.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
        paths.append((path, chunk_idx))
    return paths


def object_to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"value": repr(obj)}


def save_batch_error(client, batch, args):
    details = object_to_dict(batch)
    error_file_id = details.get("error_file_id")
    error_path = os.path.join(args.work_dir, f"{batch.id}_errors.jsonl")
    if error_file_id:
        response = client.files.content(error_file_id)
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        return error_path

    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2, default=str)
    return error_path


def wait_for_batch(client, batch_id, args):
    last_status = None
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status != last_status:
            print(f"Batch {batch_id}: {batch.status}")
            last_status = batch.status
        if batch.status == "completed":
            return batch.output_file_id
        if batch.status in {"failed", "cancelled", "expired"}:
            error_path = save_batch_error(client, batch, args)
            raise RuntimeError(f"Batch {batch_id} ended with status {batch.status}. Error details: {error_path}")
        time.sleep(30)


def parse_batch_output(response, output_path):
    outputs = []
    with open(output_path, "w", encoding="utf-8") as f:
        for line in response.text.splitlines():
            if line.strip():
                f.write(line + "\n")
                outputs.append(json.loads(line))
    return outputs


def run_batches_and_save(client, input_paths, args, file_prefix, frame, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    for path, chunk_idx in input_paths:
        print(f"Uploading chunk {chunk_idx}: {path}")
        with open(path, "rb") as f:
            input_file = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"{file_prefix} gpt_score chunk {chunk_idx}"},
        )
        output_file_id = wait_for_batch(client, batch.id, args)
        response = client.files.content(output_file_id)

        chunk_output_path = os.path.join(args.work_dir, f"{file_prefix}_chunk_{chunk_idx}_output.jsonl")
        outputs = parse_batch_output(response, chunk_output_path)
        frame = apply_outputs(frame, outputs, args.output_col)
        frame.to_csv(output_path, encoding="utf-8", index=False)
        print(f"Saved chunk {chunk_idx} results to {output_path}")
    return frame


def extract_content(output):
    try:
        return output["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return "ERROR"


def apply_outputs(frame, outputs, output_col):
    if output_col not in frame.columns:
        frame[output_col] = pd.NA
    for output in outputs:
        custom_id = output.get("custom_id")
        if custom_id is None:
            continue
        row_idx = int(custom_id)
        frame.at[row_idx, output_col] = extract_content(output)
    return frame


def judge_one_csv(client, input_path, args, file_prefix):
    output_path = output_csv_path(args, input_path)
    if os.path.isfile(output_path) and not args.force:
        frame = load_response_csv(output_path)
        print(f"Resuming from existing output CSV: {output_path}")
    else:
        frame = load_response_csv(input_path)

    row_indices = rows_to_judge(frame, args)
    if not row_indices:
        print(f"Skipping complete file: {input_path}")
        return

    input_paths = write_batch_inputs(frame, row_indices, args, file_prefix)
    print(f"Prepared {len(row_indices)} judge requests in {len(input_paths)} chunk files.")
    if args.create_only:
        return

    run_batches_and_save(client, input_paths, args, file_prefix, frame, output_path)
    print(f"Saved judged CSV to {output_path}")


def should_skip_model(model):
    model_name = MODEL_NAME_MAP.get(model, model)
    return "ministral" in model.lower() or "ministral" in model_name.lower()


def build_jobs(args):
    if args.csv is not None:
        return [(args.csv, args.file_prefix or os.path.splitext(os.path.basename(args.csv))[0])]

    datanames = DEFAULT_DATASETS if args.dataname is None else [args.dataname]
    splits = ["train", "test"] if args.split is None else [args.split]
    models = list(MODEL_NAME_MAP.keys()) if args.model is None else [args.model]

    jobs = []
    for model in models:
        if should_skip_model(model):
            print(f"Skipping Ministral model: {model}")
            continue
        for dataname in datanames:
            for split in splits:
                path = default_csv_path(args, dataname, model, split)
                if not os.path.isfile(path):
                    print(f"Skipping missing CSV: {path}")
                    continue
                prefix = f"{dataname}_{model}_{split}_{args.experiment}_{args.prompt_name}_gpt_score"
                jobs.append((path, prefix))
    return jobs


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Judge one explicit new-format CSV.")
    parser.add_argument("--model", default=None, help="Defaults to all non-Ministral model keys in generation.models.MODEL_NAME_MAP.")
    parser.add_argument("--dataname", default=None, help="Defaults to nq, sciq, triviaqa, popqa.")
    parser.add_argument("--split", default=None, choices=["train", "test"], help="Defaults to both train and test.")
    parser.add_argument("--experiment", default="greedy")
    parser.add_argument("--prompt_name", default=REFERENCE_PROMPT)
    parser.add_argument("--response_dir", default=DEFAULT_RESPONSE_DIR)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rejudge rows even if gpt_score already exists.")
    parser.add_argument("--question_col", default="question")
    parser.add_argument("--target_col", default="target")
    parser.add_argument("--response_col", default="response")
    parser.add_argument("--output_col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--key", default="openai_key")
    parser.add_argument("--key_file", default="key.yml", help="YAML file with the API key; falls back to $OPENAI_API_KEY if absent.")
    parser.add_argument("--judge_model", default="gpt-4o-2024-11-20")
    parser.add_argument("--chunk_size", default=250, type=int)
    parser.add_argument("--max_tokens", default=550, type=int)
    parser.add_argument("--work_dir", default=None)
    parser.add_argument("--file_prefix", default=None)
    parser.add_argument("--create_only", action="store_true")
    args = parser.parse_args(argv)

    if args.work_dir is None:
        args.work_dir = os.path.join(args.response_dir, "batch_gpt_score")
    return args


def make_client(args):
    import yaml
    from openai import OpenAI

    if os.path.isfile(args.key_file):
        with open(args.key_file, encoding="utf-8") as f:
            return OpenAI(api_key=yaml.safe_load(f)[args.key])
    return OpenAI()  # reads OPENAI_API_KEY


def main(argv=None):
    args = parse_args(argv)
    jobs = build_jobs(args)
    if not jobs:
        print("No CSV files to judge.")
        return

    # A cluster-provided SSL_CERT_FILE can break the OpenAI client's TLS setup.
    os.environ.pop("SSL_CERT_FILE", None)
    client = None if args.create_only else make_client(args)

    for input_path, file_prefix in jobs:
        print("=" * 80)
        print(f"Judging: {input_path}")
        judge_one_csv(client, input_path, args, file_prefix)


if __name__ == "__main__":
    main()
