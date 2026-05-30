# MedNLP Platform

Plateforme RAG clinique pour l'analyse sémantique de notes médicales. Ingestion de documents FHIR, pipeline NLP (segmentation → déidentification → NER → vectorisation), recherche sémantique et génération augmentée par retrieval pour les cliniciens.

---

## Architecture

```
GCP Pub/Sub
    └── Ingestion FHIR (DocumentReference / DiagnosticReport)
            └── Pipeline Prefect
                    ├── Segmentation
                    ├── Déidentification (Presidio)
                    ├── NER médical (spaCy + custom)
                    ├── Quality gate
                    └── Vectorisation (BiomedBERT / LoRA-Mistral)
                            └── Qdrant
                                    └── RAG (MedCPT reranker + OpenRouter LLM)
                                            └── API FastAPI → Portal Streamlit
```

| Couche | Technologie |
|--------|-------------|
| Ingestion | GCP Pub/Sub, FHIR R4 |
| Pipeline | Prefect, Presidio, spaCy |
| Embeddings | BiomedBERT, LoRA-Mistral, routeur A/B |
| Vector store | Qdrant |
| RAG | MedCPT reranker, OpenRouter (Claude 3.5 Sonnet) |
| API | FastAPI, JWT, Prometheus |
| Workers | Celery + Redis |
| Portal | Streamlit |
| Déploiement | Cloud Run (GCP), GitHub Actions |
| Fine-tuning | Vertex AI |

---

## Démarrage rapide

### Prérequis

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)
- Docker & Docker Compose

### Installation

```bash
git clone https://github.com/SeydinaBANE/mednlp-platform.git
cd mednlp-platform

cp .env.example .env       # renseigner les clés API
make install               # uv sync + pre-commit hooks
make dev                   # démarre tous les services Docker
```

### Services locaux

| Service | URL |
|---------|-----|
| API FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Portal Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

---

## Commandes principales

```bash
make test             # Tests unitaires (couverture ≥ 80%)
make test-integration # Tests intégration (nécessite Docker)
make lint             # ruff + mypy strict
make migrate          # Migrations Alembic
make security-scan    # Bandit SAST + Safety CVEs
make eval-rag         # Évaluation RAG offline (RAGAS)
make load-test        # Locust 50 users × 5 min
make fine-tune TASK=icd10   # Fine-tuning sur Vertex AI
make backfill MODEL=v2      # Re-vectorisation du corpus
```

---

## Déploiement

| Environnement | Déclencheur |
|---------------|-------------|
| Staging | Automatique sur merge `main` |
| Production | Manuel via `gh workflow run` + approbation |

Voir [`docs/deployment.md`](docs/deployment.md) pour le guide complet de configuration GCP.

---

## Tests

```bash
# Un test isolé
uv run pytest tests/unit/test_fhir_parser.py -v

# Tous les tests
make test-all
```

Couverture minimum : **80%** sur `src/`. Les fixtures utilisent des notes synthétiques MIMIC-like — aucune PHI réelle.

---

## Contribuer

1. Créer une branche depuis `develop`
2. Commits en [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, `docs:`, `test:`)
3. PR vers `develop` — le CI doit être vert (lint + tests + scan Trivy)
4. Merge vers `main` déclenche le CD staging automatiquement

---

## Licence

Propriétaire — usage interne uniquement.
