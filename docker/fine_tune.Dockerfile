FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y python3.11 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
# Installe avec dépendances GPU
RUN uv sync --frozen --no-install-project --no-dev --extra gpu

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y python3.11 && rm -rf /var/lib/apt/lists/*
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

USER appuser

# Point d'entrée variable selon le job Vertex AI (AIP_TRAINING_DATA_URI etc.)
ENTRYPOINT ["python", "-m"]
CMD ["src.fine_tuning.vertex_job"]
