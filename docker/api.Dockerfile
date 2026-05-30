FROM python:3.11-slim AS builder

WORKDIR /app

# uv — binaire statique, aucune dépendance système
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# ── Runtime ────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Upgrade system-level pip/wheel/setuptools to fix CVE-2026-24049 (wheel<0.46.2)
# and CVE-2026-23949 (jaraco.context<6.1.0 vendored inside setuptools<76).
# These packages ship with the base image and are not managed by the uv venv.
RUN pip install --no-cache-dir --upgrade "wheel>=0.46.2" "setuptools>=76.0.0"

# Non-root user — sécurité obligatoire
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copie uniquement le venv construit dans le builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
