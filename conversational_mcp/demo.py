from __future__ import annotations

import asyncio

from .examples.productivity import build_engine


async def main() -> None:
    engine = build_engine()
    response = await engine.start("demo-user", "create task")
    print(response.message)
    response = await engine.continue_session(
        response.session_id or "", "title=Write portfolio README, due_date=2026-08-20"
    )
    print(response.message)


if __name__ == "__main__":
    asyncio.run(main())
