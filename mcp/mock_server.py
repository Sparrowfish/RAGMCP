"""A safe MCP server with mock tools used to demonstrate the package publicly."""
from datetime import datetime
from uuid import uuid4
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("conversational-mcp-demo")


@mcp.tool()
async def create_task(title: str, due_date: str | None = None, priority: str = "normal") -> dict:
    return {"status": "completed", "message": "Task created.", "data": {"id": str(uuid4()), "title": title, "due_date": due_date, "priority": priority}}


@mcp.tool()
async def create_calendar_event(title: str, start: datetime, duration_minutes: int = 30, attendees: list[str] | None = None) -> dict:
    return {"status": "completed", "message": "Calendar event created.", "data": {"id": str(uuid4()), "title": title, "start": start.isoformat(), "duration_minutes": duration_minutes, "attendees": attendees or []}}


@mcp.tool()
async def send_email(to: list[str], subject: str, body: str) -> dict:
    return {"status": "completed", "message": "Email queued (mock).", "data": {"id": str(uuid4()), "to": to, "subject": subject, "body": body}}


if __name__ == "__main__":
    mcp.run()
