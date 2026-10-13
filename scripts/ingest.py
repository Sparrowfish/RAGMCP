# scripts/ingest.py
"""
Ingest documents into the vector store.
Usage: python scripts/ingest.py --input data/raw/
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.rag.pipeline import ingest


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument("--input", default="data/raw/", help="Directory with source documents")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(cfg["logging"]["level"], cfg["logging"]["log_dir"])

    vs = ingest(args.input, cfg)
    print(f"\n✅ Done. {vs.index.ntotal} vectors stored.")


if __name__ == "__main__":
    main()