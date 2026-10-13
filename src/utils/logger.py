# src/utils/logger.py
import sys
from pathlib import Path
from loguru import logger

def setup_logger(log_level: str = "INFO", log_dir: str = "./logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.remove()  # remove default handler

    # Console — clean and readable
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File — full detail for debugging
    logger.add(
        Path(log_dir) / "app_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="7 days",
        compression="zip",
    )

logger.add  # re-export