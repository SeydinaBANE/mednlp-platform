# TODO — MedNLP Platform

## Phase 1 — Foundation (Semaines 1-2)

### Tooling & DevOps
- [x] `pyproject.toml` — PEP 621, uv, ruff, mypy, pytest config
- [x] `.pre-commit-config.yaml` — ruff, mypy, bandit, commitizen
- [x] `Makefile` — toutes les commandes dev
- [x] `.env.example` — template variables d'environnement
- [x] `.gitignore`
- [x] `CLAUDE.md` — guide codebase
- [x] `TODO.md`
- [x] `docker-compose.yml` — stack dev complète
- [x] `docker/api.Dockerfile` — multi-stage, non-root
- [x] `.github/workflows/ci.yml` — lint, test, security

### Infrastructure (Terraform)
- [ ] `infra/modules/gcp_pubsub/` — topics + DLQ (5-retry)
- [ ] `infra/modules/gcp_cloudsql/` — PostgreSQL 15
- [ ] `infra/modules/gcp_gcs/` — buckets raw/processed/artifacts
- [ ] `infra/environments/staging/` + `production/`

### Core
- [x] `src/core/config.py` — Pydantic Settings v2
- [x] `src/core/openrouter_client.py` — client AsyncOpenAI → OpenRouter
- [x] `src/core/exceptions.py` — hiérarchie exceptions domaine
- [x] `src/core/telemetry.py` — OpenTelemetry setup
- [x] `src/core/models.py` — SQLAlchemy 2.0 ORM
- [ ] `src/core/schemas.py` — Pydantic v2 request/response
- [ ] Migrations Alembic initiales (`alembic init` + `alembic revision`)

### Ingestion
- [x] `src/ingestion/fhir_parser.py` — HL7 FHIR R4 → NoteRecord
- [x] `src/ingestion/pubsub_consumer.py` — pull async avec circuit-breaker
- [ ] `src/ingestion/dlq_handler.py` — handler DLQ + alerting PagerDuty
- [ ] `src/ingestion/batch_trigger.py` — GCS object.finalize → Pub/Sub

### Tests
- [ ] `tests/conftest.py` — fixtures Testcontainers
- [ ] `tests/unit/test_fhir_parser.py`
- [ ] `tests/unit/test_pubsub_consumer.py`
- [ ] `tests/integration/test_pubsub_consumer.py` — émulateur Pub/Sub
- [ ] `tests/fixtures/sample_notes/` — 5 notes synthétiques MIMIC-like

---

## Phase 2 — Vector Store + A/B Router (Semaines 3-4)

- [x] `src/vector_store/client.py` — Qdrant async client wrapper
- [x] `src/vector_store/collections.py` — gestion collections par version modèle
- [x] `src/vector_store/indexer.py` — upsert batch async + backpressure
- [x] `src/vector_store/search.py` — ANN search + filtres metadata
- [x] `src/embeddings/base_embedder.py` — interface abstraite
- [x] `src/embeddings/biomedbert_embedder.py`
- [x] `src/embeddings/lora_mistral_embedder.py` — 4-bit bitsandbytes
- [x] `src/embeddings/ab_router.py` — routage déterministe hash(note_id)
- [x] `src/embeddings/registry.py` — client MLflow model registry
- [x] Tests unit A/B router (déterminisme + dual-write)

---

## Phase 3 — RAG Query Engine (Semaines 5-6)

- [x] `src/rag/retriever.py` — ANN Qdrant ef=128
- [x] `src/rag/reranker.py` — MedCPT cross-encoder + cache Redis 10min
- [x] `src/rag/context_builder.py` — window packing (max 3500 tokens) + citation map
- [x] `src/rag/answer_generator.py` — OpenRouter streaming (Claude 3.5 Sonnet)
- [x] `src/rag/guardrails.py` — re-scan PHI + disclaimers cliniques
- [x] `src/rag/prompt_templates.py` — templates versionnés depuis config YAML
- [x] `src/api/routers/query.py` — POST /query + /query/stream (SSE)
- [x] Tests unit retriever, reranker, guardrails
- [x] `scripts/evaluate_rag.py` — RAGAS offline (faithfulness ≥ 0.80)

---

## Phase 4 — Fine-Tuning LoRA (Semaines 7-8)

- [x] `src/fine_tuning/data_prep.py` — MinHash dedup + stratified split 70/15/15
- [x] `src/fine_tuning/lora_config.py` — configs PEFT (r=16, alpha=32)
- [x] `src/fine_tuning/icd10_trainer.py` — BCEWithLogitsLoss multi-label, 50 codes
- [x] `src/fine_tuning/triage_trainer.py` — 5-class ESI, weighted-F1
- [x] `src/fine_tuning/vertex_job.py` — CustomTrainingJob Vertex AI (T4)
- [x] `src/fine_tuning/evaluator.py` — macro-F1, AUC-ROC, bootstrap CI
- [x] `src/fine_tuning/promote_model.py` — gate MLflow → Staging/Production
- [x] `docker/fine_tune.Dockerfile` — CUDA 12.1 (existait déjà)
- [x] `.github/workflows/fine-tune-trigger.yml` — dispatch manuel

---

## Phase 5 — Drift + MLOps (Semaines 9-10)

- [x] `src/drift_detection/embedding_drift.py` — JS divergence + NDArray typing
- [x] `src/drift_detection/label_drift.py` — KS test ICD-10 (Bonferroni)
- [x] `src/drift_detection/data_drift.py` — NER entity counts (chi-squared)
- [x] `src/drift_detection/alert_publisher.py` — PagerDuty + Slack
- [x] `src/core/telemetry.py` — Prometheus Gauges drift (embedding/label/data)
- [x] Dashboards Grafana (`config/grafana/dashboard_mednlp.json`)
- [x] Alerting Cloud Monitoring (`config/alerting_policies.yaml`)

---

## Phase 6 — Portal + Hardening + Production (Semaines 11-12)

- [x] `src/portal/pages/01_semantic_search.py` — RAG query UI Streamlit
- [x] `src/portal/pages/04_model_dashboard.py` — drift + A/B charts + MLflow
- [x] `src/api/middleware.py` — JWT HS256, rate limiting Redis (sliding window), OTel tracing
- [x] `src/api/routers/audit.py` — GET /audit/{note_id} + GET /audit/actor/{actor}
- [x] `src/core/database.py` — async SQLAlchemy session factory
- [x] `tests/load/locustfile.py` — Locust p95 SLA assertions (ingest < 500ms, RAG < 3s)
- [x] Security : Bandit + Safety + Trivy déjà dans `.github/workflows/ci.yml`
- [x] VPC/CMEK/Secret Manager : `config/alerting_policies.yaml` (Cloud Monitoring)
- [x] `docs/runbook.md` — incident response, rollback, scaling
- [x] `.github/workflows/cd-staging.yml` — deploy automatique sur merge main
- [x] `.github/workflows/cd-production.yml` — deploy manuel + canary 5%
