"""Pack ranked search results into a token-bounded context string with citation map."""

from dataclasses import dataclass, field
from datetime import datetime

import structlog

from src.core.schemas import CitationSource
from src.ingestion.schemas import NoteType
from src.vector_store.search import SearchResult

logger = structlog.get_logger(__name__)

_MAX_TOKENS = 3500
_MAX_CHARS_PER_NOTE = 1200  # rough guard before token counting
_CHARS_PER_TOKEN = 4  # approximation for Claude models


@dataclass
class BuiltContext:
    context_text: str
    citations: list[CitationSource]
    total_tokens: int
    citation_map: dict[int, str] = field(default_factory=dict)  # index → note_id


def build_context(
    results: list[SearchResult],
    max_tokens: int = _MAX_TOKENS,
) -> BuiltContext:
    """Pack search results into a context string bounded by max_tokens.

    Each note gets a citation index [1], [2], ... appended to the context.
    Notes that would exceed the token budget are dropped.
    """
    lines: list[str] = []
    citations: list[CitationSource] = []
    citation_map: dict[int, str] = {}
    total_tokens = 0

    for idx, result in enumerate(results, start=1):
        excerpt = _extract_excerpt(result)
        entry = f"[{idx}] {excerpt}"
        entry_tokens = _count_tokens(entry)

        if total_tokens + entry_tokens > max_tokens:
            logger.debug("context_budget_reached", idx=idx, total=total_tokens)
            break

        lines.append(entry)
        total_tokens += entry_tokens
        citation_map[idx] = result.note_id

        citations.append(
            CitationSource(
                note_id=result.note_id,
                patient_id=str(result.payload.get("patient_id", "")),
                note_type=_parse_note_type(result.payload.get("note_type")),
                authored_at=_parse_date(result.payload.get("authored_at")),
                excerpt=excerpt[:300],
                score=result.score,
            )
        )

    context_text = "\n\n".join(lines)
    logger.info(
        "context_built",
        notes_included=len(lines),
        total_tokens=total_tokens,
        notes_available=len(results),
    )

    return BuiltContext(
        context_text=context_text,
        citations=citations,
        total_tokens=total_tokens,
        citation_map=citation_map,
    )


def _extract_excerpt(result: SearchResult) -> str:
    text = result.payload.get("processed_text") or result.payload.get("raw_text") or ""
    return text[:_MAX_CHARS_PER_NOTE].strip()


def _count_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _parse_note_type(value: object) -> NoteType:
    try:
        return NoteType(str(value))
    except ValueError:
        return NoteType.UNKNOWN


def _parse_date(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        from datetime import UTC

        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        from datetime import UTC

        return datetime.now(tz=UTC)
