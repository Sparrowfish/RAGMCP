"""Safe, domain-neutral normalizers applied before Pydantic validation."""
from datetime import datetime
from typing import Any


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def normalize_date(value: Any) -> str:
    text = normalize_text(value)
    return datetime.fromisoformat(text).date().isoformat()


def normalize_datetime(value: Any) -> str:
    return datetime.fromisoformat(normalize_text(value).replace("Z", "+00:00")).isoformat()


def normalize_email_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else str(value).split(",")
    return [normalize_text(item).lower() for item in values if normalize_text(item)]


def normalize_arguments(arguments: dict[str, Any], normalizers: dict[str, Any]) -> dict[str, Any]:
    return {key: normalizers[key](value) if key in normalizers else value for key, value in arguments.items()}
