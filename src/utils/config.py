# src/utils/config.py
import yaml
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(Path(config_path), "r") as f:
        config = yaml.safe_load(f)
    return config

def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set in .env file")
    return token