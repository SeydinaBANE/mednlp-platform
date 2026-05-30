.DEFAULT_GOAL := help

.PHONY: help install dev lint lint-fix test test-integration test-all load-test \
        migrate migrate-down bootstrap docker-build docker-up docker-down docker-logs \
        clean fine-tune eval-rag deploy-staging deploy-prod security-scan

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────

install:  ## Installe les dépendances + hooks pre-commit
	uv sync --all-extras
	uv run pre-commit install --install-hooks

update:  ## Met à jour le lockfile
	uv lock --upgrade

# ── Dev ────────────────────────────────────────────────────────────────────────

dev:  ## Lance tous les services locaux (Docker Compose)
	docker compose up -d
	@echo "Services démarrés. Logs: make logs"

logs:  ## Affiche les logs de l'API
	docker compose logs -f api

shell:  ## Ouvre un shell dans le container API
	docker compose exec api bash

# ── Qualité ────────────────────────────────────────────────────────────────────

lint:  ## Vérifie ruff + mypy
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/

lint-fix:  ## Corrige automatiquement les erreurs ruff
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

security-scan:  ## Bandit (SAST) + Safety (CVEs)
	uv run bandit -r src/ -ll --skip B101 -f json -o reports/bandit.json
	uv run safety check --full-report
	@echo "Rapport Bandit : reports/bandit.json"

# ── Tests ──────────────────────────────────────────────────────────────────────

test:  ## Tests unitaires avec couverture (≥ 80%)
	uv run pytest tests/unit/ -v

test-integration:  ## Tests d'intégration (nécessite Docker)
	docker compose -f docker-compose.test.yml up -d
	uv run pytest tests/integration/ -v --timeout=120 --no-cov
	docker compose -f docker-compose.test.yml down -v

test-all:  ## Tous les tests (unit + integration)
	make test
	make test-integration

load-test:  ## Load test Locust — 50 users, 5 minutes
	uv run locust -f tests/load/locustfile.py --headless \
		-u 50 -r 5 --run-time 5m --host http://localhost:8000

eval-rag:  ## Évaluation RAG offline (RAGAS)
	uv run python scripts/evaluate_rag.py \
		--output reports/rag_eval_$$(date +%Y%m%d_%H%M%S).json

# ── Base de données ────────────────────────────────────────────────────────────

migrate:  ## Applique les migrations Alembic
	uv run alembic upgrade head

migrate-down:  ## Rollback d'une migration
	uv run alembic downgrade -1

bootstrap:  ## Initialise la DB + données de config
	uv run python scripts/bootstrap_db.py

# ── Docker ─────────────────────────────────────────────────────────────────────

docker-build:  ## Build toutes les images sans cache
	docker compose build --no-cache

docker-up:  ## Démarre tous les services
	docker compose up -d

docker-down:  ## Arrête et supprime les containers + volumes
	docker compose down -v

# ── ML ─────────────────────────────────────────────────────────────────────────

fine-tune:  ## Lance un job de fine-tuning sur Vertex AI (TASK=icd10|triage)
	@test -n "$(TASK)" || (echo "Usage: make fine-tune TASK=icd10" && exit 1)
	uv run python scripts/fine_tune_trigger.py --task $(TASK)

backfill:  ## Re-vectorise tout le corpus pour un nouveau modèle (MODEL=v2)
	@test -n "$(MODEL)" || (echo "Usage: make backfill MODEL=v2" && exit 1)
	uv run python scripts/backfill_vectors.py --model $(MODEL)

# ── Déploiement ────────────────────────────────────────────────────────────────

deploy-staging:  ## Déploie sur staging via GitHub Actions
	gh workflow run cd-staging.yml

deploy-prod:  ## Déploie en production (gate manuel)
	gh workflow run cd-production.yml

# ── Nettoyage ──────────────────────────────────────────────────────────────────

clean:  ## Supprime les artefacts (cache, .pyc, rapports)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov/ dist/
	@echo "Artefacts supprimés."
