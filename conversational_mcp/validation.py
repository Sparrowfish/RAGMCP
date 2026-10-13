from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, ValidationError


def missing_fields(model: Type[BaseModel], values: dict[str, Any]) -> list[str]:
    """Return required fields not supplied yet, without requiring a full model."""
    missing = []
    for name, field in model.model_fields.items():
        if name not in values and field.is_required():
            missing.append(name)
    return missing


def validate_complete(model: Type[BaseModel], values: dict[str, Any]) -> dict[str, Any]:
    try:
        return model.model_validate(values).model_dump()
    except ValidationError as exc:
        raise ValueError(exc.errors()) from exc
