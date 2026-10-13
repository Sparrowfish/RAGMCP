# src/model/loader.py
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from loguru import logger
from src.utils.config import load_config, get_hf_token


def get_bnb_config(cfg: dict) -> BitsAndBytesConfig | None:
    """Build bitsandbytes quantization config from settings."""
    if not cfg["model"].get("load_in_4bit", True):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — best quality
        bnb_4bit_compute_dtype=torch.float16, # compute in fp16, not fp32
        bnb_4bit_use_double_quant=True,       # nested quantization saves ~0.4GB
    )


def load_tokenizer(cfg: dict) -> AutoTokenizer:
    model_name = cfg["model"]["name"]
    cache_dir = cfg["model"]["cache_dir"]

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=get_hf_token(),
    )

    # Mistral has no pad token by default — required for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_model(cfg: dict) -> AutoModelForCausalLM:
    model_name = cfg["model"]["name"]
    cache_dir = cfg["model"]["cache_dir"]
    device = cfg["model"]["device"]

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    bnb_config = get_bnb_config(cfg)

    logger.info(f"Loading model: {model_name} | 4-bit: {bnb_config is not None} | device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,          # "cuda" = put everything on GPU
        cache_dir=cache_dir,
        token=get_hf_token(),
        torch_dtype=torch.float16,
    )

    model.eval()  # inference mode — disables dropout
    logger.success(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model