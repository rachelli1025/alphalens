from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from alphalens_core.db.session import session_scope

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(session: Annotated[AsyncSession, Depends(session_scope)]) -> dict[str, str]:
    """Readiness: the database is reachable and pgvector is installed."""
    try:
        ext = await session.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname='vector'")
        )
    except Exception as exc:  # noqa: BLE001 - report any DB failure as not-ready
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
    if ext is None:
        raise HTTPException(status_code=503, detail="pgvector extension missing")
    return {"status": "ready", "pgvector": ext}
