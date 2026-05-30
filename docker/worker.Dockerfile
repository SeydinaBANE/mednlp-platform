FROM python:3.11-slim AS builder

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

FROM python:3.11-slim AS runtime

# Upgrade system-level pip/wheel/setuptools to fix CVE-2026-24049 (wheel<0.46.2)
# and CVE-2026-23949 (jaraco.context<6.1.0 vendored inside setuptools<76).
RUN pip install --no-cache-dir --upgrade "wheel>=0.46.2" "setuptools>=76.0.0"

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

USER appuser

CMD ["celery", "-A", "src.workers.app", "worker", "--loglevel=info", "--concurrency=4"]
