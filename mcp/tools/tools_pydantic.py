"""Example domain models. Replace these with models for your own MCP server."""
from datetime import datetime
import re
from pydantic import BaseModel, Field, field_validator

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class CreateTaskArgs(BaseModel):
    title: str = Field(min_length=3, max_length=200, description="Short task title")
    due_date: str | None = Field(default=None, description="Due date in YYYY-MM-DD format")
    priority: str = Field(default="normal", pattern="^(low|normal|high)$")


class CreateCalendarEventArgs(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    start: datetime
    duration_minutes: int = Field(default=30, ge=5, le=480)
    attendees: list[str] = Field(default_factory=list)

    @field_validator("attendees")
    @classmethod
    def validate_attendees(cls, value: list[str]) -> list[str]:
        if not all(_EMAIL.fullmatch(email) for email in value):
            raise ValueError("attendees must be valid email addresses")
        return value


class SendEmailArgs(BaseModel):
    to: list[str] = Field(min_length=1)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10_000)

    @field_validator("to")
    @classmethod
    def validate_recipients(cls, value: list[str]) -> list[str]:
        if not all(_EMAIL.fullmatch(email) for email in value):
            raise ValueError("recipients must be valid email addresses")
        return value
