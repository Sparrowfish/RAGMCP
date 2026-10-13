# scripts/test_inference.py
"""
Smoke test: load model and run a single generation.
Run with: python scripts/test_inference.py
"""
import sys
import time
from pathlib import Path

# make src/ importable from scripts/
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.model.loader import load_model, load_tokenizer
from src.model.inference import generate


def main():
    cfg = load_config()
    setup_logger(cfg["logging"]["level"], cfg["logging"]["log_dir"])

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)

    print("\n" + "="*50)
    print("Model loaded. Running test generation...")
    print("="*50 + "\n")

    start = time.time()
    response = generate(
        model=model,
        tokenizer=tokenizer,
        user_message="Explain what a transformer neural network is in 3 sentences.",
        system_prompt="You are a concise technical assistant.",
    )
    elapsed = time.time() - start

    print(f"Response:\n{response}")
    print(f"\nGeneration time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()