# MedNLP Platform — Runbook

## Services & URLs

| Service | Staging | Production |
|---------|---------|------------|
| API | `https://mednlp-api-staging-*.run.app` | `https://mednlp-api-prod-*.run.app` |
| Grafana | `http://grafana:3000` | Cloud Monitoring |
| MLflow | `http://mlflow:5000` | Vertex AI Experiments |
| Qdrant | `http://qdrant:6333` | Qdrant Cloud |

---

## Incident Response

### RAG p99 latency > 30s

1. Check Grafana panel `RAG Query Latency (p99)` — identify which stage is slow
2. Check Qdrant: `curl http://qdrant:6333/collections` — confirm collections exist
3. Check Redis reranker cache hit rate: `redis-cli info stats | grep keyspace_hits`
4. Check OpenRouter rate limits: inspect `llm_tokens_total` counter in Grafana
5. If Qdrant is slow: restart container `docker compose restart qdrant`
6. Escalate to #platform-oncall if latency persists > 10 minutes

### DLQ rate > 5%

1. Check `dlq_messages_total` by `reason` label in Grafana
2. If `reason=FHIRParseError`: inspect incoming messages in Pub/Sub console
3. If `reason=decode_error`: check upstream FHIR server encoding
4. Check logs: `make logs | grep dlq_published`
5. Drain DLQ topic manually if messages are recoverable
6. Update `_NOTE_TYPE_MAP` in `fhir_parser.py` if new LOINC codes are arriving

### Embedding drift > 0.1

1. Check `embedding_drift_score` gauge — if > 0.2, trigger backfill
2. Identify window of drift by comparing Grafana time series
3. Run: `make backfill MODEL=v2` (re-vectorise corpus with current model)
4. If drift persists, schedule fine-tuning: `make fine-tune TASK=icd10`
5. Monitor drift score after backfill — should return < 0.05 within 24h

### Pipeline failure rate > 1%

1. Check `pipeline_note_processing_total{status="failure"}` in Grafana
2. Inspect worker logs: `docker compose logs worker-nlp`
3. Common causes: OOM (increase container memory), spaCy model not loaded
4. Restart workers: `docker compose restart worker-nlp`

---

## Rollback

### Cloud Run rollback (< 2 min)

```bash
# List recent revisions
gcloud run revisions list --service mednlp-api-prod --region us-central1

# Roll back to previous revision
gcloud run services update-traffic mednlp-api-prod \
  --region us-central1 \
  --to-revisions PREVIOUS_REVISION=100
```

### Database migration rollback

```bash
make migrate-down  # rolls back one migration
# Or to a specific version:
uv run alembic downgrade <revision_id>
```

---

## Scaling

| Resource | Trigger | Action |
|----------|---------|--------|
| API CPU > 80% | 5min sustained | `--max-instances` +10 |
| Qdrant memory > 85% | 1min sustained | Upgrade node type |
| Redis memory > 90% | Alert | Evict cache keys, increase maxmemory |
| DLQ size > 1000 | Immediate | Page on-call |

---

## Key Make Commands

```bash
make logs          # Follow API logs
make shell         # Shell into API container
make migrate       # Apply pending migrations
make migrate-down  # Rollback one migration
make backfill MODEL=v2  # Re-vectorise with new model
make fine-tune TASK=icd10  # Trigger fine-tuning job
make eval-rag      # Offline RAGAS evaluation
make security-scan # Bandit + Safety CVE check
```

---

## Contacts

- On-call rotation: #platform-oncall (PagerDuty escalation policy)
- HIPAA compliance queries: security@mednlp.internal
- MLOps questions: mlops@mednlp.internal
