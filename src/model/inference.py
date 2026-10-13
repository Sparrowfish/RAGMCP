# src/model/inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from src.utils.config import load_config


# Mistral instruct prompt format
MISTRAL_PROMPT_TEMPLATE = "<s>[INST] {system}\n\n{user} [/INST]"


def build_prompt(user_message: str, system_prompt: str = "", context: str = "") -> str:
    """
    Construct the final prompt string.
    If RAG context is provided, it's injected into the system prompt.
    """
    if context:
        system_prompt = (
            f"{system_prompt}\n\n"
            f"Use the following context to answer the question. "
            f"If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}"
        )

    return MISTRAL_PROMPT_TEMPLATE.format(
        system=system_prompt.strip(),
        user=user_message.strip(),
    )


@torch.inference_mode()  # faster than torch.no_grad(), disables grad entirely
def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_message: str,
    system_prompt: str = "You are a helpful assistant.",
    context: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Run a single forward pass and return the generated text.
    All generation parameters are explicit — no magic defaults.
    """
    prompt = build_prompt(user_message, system_prompt, context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3072,  # leave room for generation within 4096 context
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    logger.debug(f"Prompt tokens: {input_len}")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Slice off the input tokens — return only the new generated tokens
    new_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    logger.debug(f"Generated tokens: {len(new_tokens)}")
    return response.strip()