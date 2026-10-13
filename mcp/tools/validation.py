"""Partial and final Pydantic validation for multi-turn tool argument collection."""
from typing import Any
from pydantic import BaseModel, ValidationError


def missing_fields(schema: type[BaseModel], arguments: dict[str, Any]) -> list[str]:
    return [name for name, field in schema.model_fields.items() if field.is_required() and arguments.get(name) is None]


def validate_partial(schema: type[BaseModel], arguments: dict[str, Any]) -> dict[str, list[dict[str, Any]] | bool]:
    errors = []
    for field_name in arguments:
        if field_name not in schema.model_fields:
            errors.append({"field": field_name, "message": "unknown_field"})
    return {"valid": not errors, "errors": errors}


def validate_complete(schema: type[BaseModel], arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        return schema.model_validate(arguments).model_dump(mode="json")
    except ValidationError as exc:
        raise ValueError(exc.errors()) from exc
