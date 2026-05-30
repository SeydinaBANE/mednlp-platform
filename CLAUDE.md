# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commandes essentielles

```bash
make install          # Setup initial (uv sync + pre-commit hooks)
make dev              # Lance tous les services locaux (Docker Compose)
make test             # Tests unitaires (couverture ≥ 80%)
make test-integration # Tests intégration (nécessite Docker)
make test-all         # Unit + integration
make lint             # ruff check + mypy strict
make lint-fix         # Corrige automatiquement les erreurs ruff
make migrate          # Applique les migrations Alembic (upgrade head)
make migrate-down     # Rollback d'une migration
make bootstrap        # Init DB + seed config
make logs             # Suit les logs de l'API (docker compose logs -f api)
make shell            # Shell interactif dans le container API
make load-test        # Locust 50 users × 5 min contre http://localhost:8000
make security-scan    # Bandit SAST + Safety CVEs → reports/bandit.json
make eval-rag         # Évaluation RAG offline (RAGAS) → reports/rag_eval_*.json
make clean            # Supprime artefacts (.ruff_cache, .mypy_cache, htmlcov, dist)
make update           # Met à jour le lockfile (uv lock --upgrade)

uv add <package>      # Ajouter une dépendance (met à jour uv.lock)
uv add --dev <pkg>    # Ajouter une dépendance de dev
uv run <cmd>          # Exécuter dans le venv sans activation manuelle
```

Lancer un test isolé :

```bash
uv run pytest tests/unit/test_fhir_parser.py -v
uv run pytest tests/unit/test_fhir_parser.py::test_parse_document_reference -v
```

Démarrer les services individuellement sans Docker :

```bash
uv run uvicorn src.api.main:app --reload --port 8000   # API FastAPI
uv run celery -A src.workers.tasks worker -l info       # Worker Celery
uv run streamlit run src/portal/pages/01_semantic_search.py  # Portal
```

## Architecture

```
src/
├── core/            — config (pydantic-settings), exceptions, OTel + Prometheus, OpenRouter client
├── ingestion/       — consommateur Pub/Sub, parser FHIR, schémas internes NoteRecord
├── pipeline/        — flow Prefect (segment → deidentify → NER → quality_gate → vectorize)
├── embeddings/      — BiomedBERT, LoRA-Mistral, routeur A/B déterministe, registre MLflow
├── vector_store/    — client Qdrant, collections, indexer batch, recherche filtrée
├── rag/             — retriever, reranker MedCPT, context builder, générateur, guardrails
├── fine_tuning/     — data prep, trainers ICD-10/triage, job Vertex AI, promotion MLflow
├── drift_detection/ — Evidently (embedding drift JSD), KS test (label drift), alerting PD/Slack
├── api/             — FastAPI : AuthMiddleware JWT + RequestTracingMiddleware, routers query/audit
├── workers/         — tâches Celery : process_note, backfill_index, trigger_fine_tune
└── portal/          — UI Streamlit clinicien (port 8501, CORS whitelisté dans l'API)
```

**Flux de données principal** : GCP Pub/Sub → `ingestion/pubsub_consumer.py` → `ingestion/fhir_parser.py` → `NoteRecord` → pipeline Prefect (`pipeline/flow.py`) → Qdrant.

### Pipeline Prefect

`process_note(note)` dans `pipeline/flow.py` exécute les stages dans l'ordre :
`segment → deidentify → NER → quality_gate → vectorize`

Chaque stage reçoit et retourne un `PipelineContext` (défini dans `pipeline/schemas.py`) qui accumule `segments`, `entities`, `quality`, `errors`, et `vector_indexed`. `skip_vectorizer=True` active le mode dry-run pour l'évaluation.

`run_batch()` utilise `asyncio.gather()` pour paralléliser sur une liste de notes.

### Ingestion Pub/Sub — modes d'échec

1. **Transitoire** (réseau, service down) → retry tenacity, puis NACK
2. **Poison pill** (FHIR malformé, `MissingPatientReferenceError`) → ACK + DLQ
3. **Timeout de traitement** → `modify_ack_deadline` (deadline=600s)

Circuit-breaker : s'ouvre après 10 échecs consécutifs, pause 60s. Parser FHIR : supporte `DocumentReference` et `DiagnosticReport`. Le mapping LOINC → `NoteType` est dans `_NOTE_TYPE_MAP`. `safe_get(obj, *keys, default=None)` est l'utilitaire de traversal dict/list.

### Embeddings — routeur A/B

Le bucket est déterministe : `MD5(note_id) % 100`. Si bucket < `traffic_b_pct × 100`, le modèle B est utilisé. Cela garantit que la même note reçoit toujours le même modèle entre ré-indexations.

`dual_write()` indexe toutes les notes avec les deux modèles simultanément — utilisé pour le backfill et les déploiements sécurisés.

Le registre `embeddings/registry.py` consomme MLflow pour résoudre le modèle en stage `Production`.

### RAG

Le retriever (`rag/retriever.py`) utilise `BiomedBertEmbedder` par défaut et sur-récupère `k=20` avant reranking MedCPT (`rag/reranker.py`). Les guardrails (`rag/guardrails.py`) s'appliquent avant et après la génération.

### Workers Celery

Broker et backend : Redis (`settings.redis_url`). Les tâches Celery sont synchrones — elles enveloppent le pipeline async avec `asyncio.run()`. Trois tâches : `tasks.process_note` (retry ×3, backoff exponentiel), `tasks.backfill_index`, `tasks.trigger_fine_tune` (retry ×1, countdown=30s).

### Hiérarchie des exceptions

`MedNLPError` est la base. Sous-classes par domaine : `IngestionError` (`FHIRParseError`, `MissingPatientReferenceError`, `DLQError`), `PipelineError` (`QualityGateError`, `DeidentificationError`), `VectorStoreError` (`CollectionNotFoundError`), `RAGError` (`GuardrailViolationError`), `LLMError`, `FineTuningError` (`ModelPromotionBlockedError`), `AuthError` (`InvalidTokenError`).

## LLM — OpenRouter

Toutes les inférences LLM passent par `src/core/openrouter_client.py`.

```python
from src.core.openrouter_client import complete, stream_complete

text = await complete(messages, model="anthropic/claude-3.5-sonnet")

async for chunk in stream_complete(messages):
    ...
```

`get_openrouter_client()` est `lru_cache` — utiliser `get_openrouter_client.cache_clear()` dans les tests qui remplacent la clé API. Modèles : `OPENROUTER_MODEL_HEAVY` (claude-3.5-sonnet, RAG clinique), `OPENROUTER_MODEL_LIGHT` (claude-3-haiku).

## Conventions de code

- Python `>=3.11,<3.13` — pas de syntaxe 3.12+ exclusive sans vérifier la contrainte
- `mypy --strict` sur tout `src/` — zéro erreur tolérée
- `ruff` lint + format — line-length 100
- Pas de `bare except:` — toujours catcher une exception spécifique
- Pas de `print()` — utiliser `structlog` uniquement
- Docker non-root obligatoire (`USER appuser`)
- Secrets via Secret Manager en prod, `.env` en dev uniquement
- Commits : Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`)

## Tests

- **Unit** (`tests/unit/`) : mocks sur toutes les dépendances externes (Qdrant, PG, OpenRouter, Pub/Sub)
- **Integration** (`tests/integration/`) : Testcontainers (PG + Redis), émulateur Pub/Sub
- **Fixtures** synthétiques dans `tests/conftest.py` et `tests/fixtures/sample_notes/` (notes MIMIC-like, jamais de vraie PHI)
- Couverture minimum : 80% sur `src/`
- `asyncio_mode = "auto"` dans `pyproject.toml` — les tests async n'ont pas besoin de `@pytest.mark.asyncio`
- `get_settings()` est `lru_cache` — utiliser `get_settings.cache_clear()` dans les tests qui surchargent les vars d'env

## Base de données

Migrations Alembic dans `alembic/versions/`. Toujours créer une migration pour tout changement de schéma :

```bash
uv run alembic revision --autogenerate -m "description"
make migrate
```

## Observabilité

Métriques Prometheus exposées via `src/core/telemetry.py` (importées directement dans chaque module qui en a besoin) :
- `pipeline_note_processing_total{status}` — notes traitées
- `pipeline_stage_latency_seconds{stage,status}` — latence par stage
- `rag_query_latency_seconds` — latence end-to-end RAG
- `embedding_model_inference_seconds{model}` — inférence embeddings
- `dlq_messages_total{reason}` — messages DLQ
- `llm_tokens_total{model,direction}` — tokens OpenRouter

En dev, `setup_telemetry()` n'expose pas Prometheus (port=None) ; en prod, il démarre un serveur sur le port 9090.

## Services locaux (Docker Compose)

| Service | URL |
|---------|-----|
| API FastAPI | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Grafana | http://localhost:3000 |
| Pub/Sub émulateur | localhost:8085 |
| Prometheus | http://localhost:9090 |
| Portal Streamlit | http://localhost:8501 |

Le service Docker `worker-stream` exécute `pubsub_consumer.py` et se connecte via `PUBSUB_EMULATOR_HOST=pubsub-emulator:8085`. L'émulateur est activé quand `PUBSUB_EMULATOR_HOST` est défini dans `.env`.

## Déploiement

- **Staging** : automatique sur merge `main` via `.github/workflows/cd-staging.yml`
- **Production** : dispatch manuel `make deploy-prod` (nécessite approbation PR)
- **Fine-tuning** : `make fine-tune TASK=icd10` → job Vertex AI (`n1-standard-8` + T4)
- **Backfill vectorisation** : `make backfill MODEL=v2` — re-vectorise tout le corpus pour un nouveau modèle
