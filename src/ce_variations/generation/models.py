"""Registry of the LLMs used for response generation and their vLLM settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    tensor_parallel: bool = False   # shard over `--num_gpu` GPUs
    limit_context: bool = False     # gpu_memory_utilization=0.9, max_model_len=4096
    mistral_format: bool = False    # Mistral-native weights, config and tokenizer


_MINISTRAL_14B = ModelSpec("mistralai/Ministral-3-14B-Reasoning-2512", True, True, True)
_MINISTRAL_8B = ModelSpec("mistralai/Ministral-3-8B-Reasoning-2512", True, True, True)

MODEL_SPECS = {
    "deepseek": ModelSpec("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", True, True),
    "qwen14b": ModelSpec("Qwen/Qwen2.5-14B-Instruct", True),
    "qwen32b": ModelSpec("Qwen/Qwen2.5-32B-Instruct", True),
    "qwen72b": ModelSpec("Qwen/Qwen2.5-72B-Instruct", True),
    "llama70b": ModelSpec("meta-llama/Llama-3.3-70B-Instruct", True, True),
    "mistral": ModelSpec("mistralai/Mistral-Large-Instruct-2411", True, True),
    "ministral14b-reasoning": _MINISTRAL_14B,
    "ministral14b": _MINISTRAL_14B,
    "ministral8b-reasoning": _MINISTRAL_8B,
    "ministral8b": _MINISTRAL_8B,
    "olmo-1": ModelSpec("allenai/OLMo-7B-Instruct-hf"),
    "olmo-1-sft": ModelSpec("allenai/OLMo-7B-SFT-hf"),
    "olmo": ModelSpec("allenai/OLMo-2-1124-7B-Instruct"),
    "olmo-sft": ModelSpec("allenai/OLMo-2-1124-7B-SFT"),
    "olmo-dpo": ModelSpec("allenai/OLMo-2-1124-7B-DPO"),
    "olmo13b": ModelSpec("allenai/OLMo-2-1124-13B-Instruct"),
    "olmo13b-sft": ModelSpec("allenai/OLMo-2-1124-13B-SFT"),
    "olmo13b-dpo": ModelSpec("allenai/OLMo-2-1124-13B-DPO"),
    "olmo32b": ModelSpec("allenai/OLMo-2-0325-32B-Instruct", True),
    "llama8b-sft": ModelSpec("allenai/Llama-3.1-Tulu-3-8B-SFT"),
    "llama8b": ModelSpec("allenai/Llama-3.1-Tulu-3-8B"),
    "llama8b-dpo": ModelSpec("allenai/Llama-3.1-Tulu-3-8B-DPO"),
}

# Short model key -> Hugging Face model id.
MODEL_NAME_MAP = {key: spec.hf_id for key, spec in MODEL_SPECS.items()}


def load_model(model, num_gpu=1, cache_dir=None):
    """Load `model` (a key of MODEL_SPECS) with vLLM; returns (llm, tokenizer)."""
    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(MODEL_SPECS)}")
    # Imported lazily so the rest of the package works without vLLM installed.
    from transformers import AutoTokenizer
    from vllm import LLM

    spec = MODEL_SPECS[model]
    kwargs = {"model": spec.hf_id, "download_dir": cache_dir}
    if spec.tensor_parallel:
        kwargs["tensor_parallel_size"] = num_gpu
    if spec.limit_context:
        kwargs.update(gpu_memory_utilization=0.9, max_model_len=4096)
    if spec.mistral_format:
        kwargs.update(tokenizer_mode="mistral", config_format="mistral", load_format="mistral")

    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer() if spec.mistral_format else AutoTokenizer.from_pretrained(spec.hf_id)
    return llm, tokenizer
