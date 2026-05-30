"""HIPAA audit trail router — GET /audit/{note_id}, POST /audit."""

from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.models import AuditLog
from src.core.schemas import AuditLogResponse, PaginatedResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get(
    "/note/{note_id}",
    response_model=PaginatedResponse,
    summary="HIPAA audit trail for a note",
)
async def get_note_audit(
    note_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse:
    """Return paginated audit log entries for a given note_id."""
    offset = (page - 1) * page_size

    stmt = (
        select(AuditLog)
        .where(AuditLog.resource_type == "note", AuditLog.resource_id == note_id)
        .order_by(AuditLog.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    count_stmt = select(AuditLog).where(
        AuditLog.resource_type == "note", AuditLog.resource_id == note_id
    )
    count_result = await db.execute(count_stmt)
    total = len(count_result.scalars().all())

    items = [AuditLogResponse.model_validate(row) for row in rows]
    return PaginatedResponse(total=total, page=page, page_size=page_size, items=items)


@router.post("", status_code=status.HTTP_201_CREATED, summary="Record an audit event")
async def record_audit(
    actor: str,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    ip_address: str | None = None,
    details: dict[str, Any] | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Create a new audit log entry. Used internally by other endpoints."""
    entry = AuditLog(
        actor=actor,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        details=details or {},
        created_at=datetime.now(tz=UTC),
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    logger.info("audit_recorded", actor=actor, action=action, resource=resource_type)
    return {"id": entry.id, "created_at": entry.created_at.isoformat()}


@router.get(
    "/actor/{actor}",
    response_model=PaginatedResponse,
    summary="Audit log by actor",
)
async def get_actor_audit(
    actor: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse:
    """Return paginated audit log entries for a given actor (user or service)."""
    offset = (page - 1) * page_size

    stmt = (
        select(AuditLog)
        .where(AuditLog.actor == actor)
        .order_by(AuditLog.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    count_result = await db.execute(select(AuditLog).where(AuditLog.actor == actor))
    total = len(count_result.scalars().all())

    items = [AuditLogResponse.model_validate(row) for row in rows]
    return PaginatedResponse(total=total, page=page, page_size=page_size, items=items)
