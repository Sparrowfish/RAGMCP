# src/finetune/dataset.py
import json
from pathlib import Path
from datasets import Dataset
from loguru import logger


# Mistral instruct format — must exactly match inference prompt format
PROMPT_TEMPLATE = """<s>[INST] {system}

{instruction}{input_block} [/INST] {output}</s>"""

SYSTEM_PROMPT = "You are a helpful, accurate assistant."


def _format_example(example: dict) -> dict:
    """Convert a raw JSONL example into a formatted training string."""
    input_block = f"\n\nInput:\n{example['input']}" if example.get("input", "").strip() else ""

    text = PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        instruction=example["instruction"].strip(),
        input_block=input_block,
        output=example["output"].strip(),
    )
    return {"text": text}


def load_jsonl_dataset(jsonl_path: str) -> Dataset:
    """Load a JSONL file and return a HuggingFace Dataset."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {i+1}: {e}")

    if not records:
        raise ValueError(f"No valid records found in {jsonl_path}")

    logger.info(f"Loaded {len(records)} training examples from {jsonl_path}")

    dataset = Dataset.from_list(records)
    dataset = dataset.map(_format_example, remove_columns=dataset.column_names)

    logger.info(f"Sample formatted example:\n{dataset[0]['text'][:300]}...")
    return dataset


def split_dataset(dataset: Dataset, eval_ratio: float = 0.1) -> tuple[Dataset, Dataset]:
    """Split into train / eval sets."""
    split = dataset.train_test_split(test_size=eval_ratio, seed=42)
    logger.info(f"Train: {len(split['train'])} | Eval: {len(split['test'])} examples")
    return split["train"], split["test"]