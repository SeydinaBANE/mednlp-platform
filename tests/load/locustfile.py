"""Locust load tests — p95 ingest < 500ms, p95 RAG < 3s.

Run:
    uv run locust -f tests/load/locustfile.py --headless \
        -u 50 -r 5 --run-time 5m --host http://localhost:8000
"""

import random

from locust import HttpUser, between, events, task

_SAMPLE_QUERIES = [
    "What medications is the patient currently taking?",
    "What is the patient's chief complaint?",
    "What diagnostic tests were ordered?",
    "What is the patient's blood pressure?",
    "Are there any drug allergies documented?",
]

_SAMPLE_PATIENT_IDS = [f"synth-patient-{i:03d}" for i in range(1, 11)]

_SAMPLE_NOTE = {
    "patient_id": "synth-patient-001",
    "note_type": "progress_note",
    "text": (
        "65yo M with HTN, DM2 presenting for routine follow-up. "
        "BP 138/84, HR 72. A1c 7.2%. Continue metformin 1000mg BID, lisinopril 10mg QD."
    ),
}

_JWT_TOKEN = "test-token"  # replaced by real token in staging/prod


class IngestUser(HttpUser):
    """Simulates clinical note ingestion — target p95 < 500ms."""

    wait_time = between(0.5, 2.0)
    weight = 3

    @task
    def ingest_note(self) -> None:
        note = dict(_SAMPLE_NOTE)
        note["patient_id"] = random.choice(_SAMPLE_PATIENT_IDS)
        note["text"] += f" Visit #{random.randint(1, 100)}."

        with self.client.post(
            "/ingest",
            json=note,
            headers={"Authorization": f"Bearer {_JWT_TOKEN}"},
            catch_response=True,
            name="POST /ingest",
        ) as resp:
            if resp.status_code == 201:
                resp.success()
            elif resp.status_code == 401:
                resp.failure("Unauthorized — check JWT token")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")


class QueryUser(HttpUser):
    """Simulates RAG queries — target p95 < 3000ms."""

    wait_time = between(1.0, 4.0)
    weight = 2

    @task(3)
    def rag_query(self) -> None:
        query = random.choice(_SAMPLE_QUERIES)
        payload = {
            "query": query,
            "top_k": 5,
        }

        with self.client.post(
            "/query",
            json=payload,
            headers={"Authorization": f"Bearer {_JWT_TOKEN}"},
            catch_response=True,
            name="POST /query",
            timeout=10,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "answer" in data:
                    resp.success()
                else:
                    resp.failure("Missing 'answer' in response")
            elif resp.status_code == 401:
                resp.failure("Unauthorized")
            elif resp.status_code == 503:
                resp.failure("No collections indexed")
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def health_check(self) -> None:
        with self.client.get("/health", catch_response=True, name="GET /health") as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")


# ── SLA assertions on test completion ─────────────────────────────────────────


@events.quitting.add_listener
def assert_sla(environment: object, **_: object) -> None:
    """Fail the load test if p95 SLAs are violated."""
    from locust.env import Environment

    assert isinstance(environment, Environment)
    stats = environment.runner.stats if environment.runner else None
    if stats is None:
        return

    violations = []
    for name, entry in stats.entries.items():
        p95 = entry.get_response_time_percentile(0.95)
        if "ingest" in str(name).lower() and p95 > 500:
            violations.append(f"Ingest p95={p95}ms > 500ms SLA")
        if "query" in str(name).lower() and p95 > 3000:
            violations.append(f"Query p95={p95}ms > 3000ms SLA")

    if violations:
        print("\n⚠️  SLA VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        environment.process_exit_code = 1
