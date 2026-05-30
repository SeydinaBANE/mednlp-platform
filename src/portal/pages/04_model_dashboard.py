"""Streamlit page — Model Drift & A/B Routing Dashboard."""

import streamlit as st

st.set_page_config(page_title="Model Dashboard", page_icon="📊", layout="wide")

st.title("📊 Model Dashboard")
st.caption("Embedding drift · Label drift · A/B routing split · Promotion history")


# ── Live metrics from Prometheus ──────────────────────────────────────────────

import httpx


@st.cache_data(ttl=30)
def _fetch_prometheus(query: str, prometheus_url: str = "http://localhost:9090") -> float | None:
    try:
        resp = httpx.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=5.0,
        )
        data = resp.json()
        result = data.get("data", {}).get("result", [])
        if result:
            return float(result[0]["value"][1])
    except Exception:  # noqa: BLE001
        pass
    return None


# ── Top metrics row ───────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

embedding_drift = _fetch_prometheus("embedding_drift_score")
label_drift_codes = _fetch_prometheus("label_drift_codes_total")
data_drift_pvalue = _fetch_prometheus("data_drift_pvalue")
dlq_rate = _fetch_prometheus("rate(dlq_messages_total[5m])")

with col1:
    val = f"{embedding_drift:.4f}" if embedding_drift is not None else "N/A"
    delta_color = "normal" if embedding_drift is None or embedding_drift < 0.1 else "inverse"
    st.metric("Embedding Drift (JS)", val, delta="threshold: 0.1", delta_color=delta_color)

with col2:
    val2 = str(int(label_drift_codes)) if label_drift_codes is not None else "N/A"
    st.metric("Drifted ICD-10 Codes", val2)

with col3:
    val3 = f"{data_drift_pvalue:.4f}" if data_drift_pvalue is not None else "N/A"
    st.metric("Data Drift p-value", val3)

with col4:
    val4 = f"{dlq_rate:.3f}/s" if dlq_rate is not None else "N/A"
    st.metric("DLQ Rate", val4)

st.divider()


# ── MLflow model registry ─────────────────────────────────────────────────────

st.subheader("Registered Models")

mlflow_uri = st.sidebar.text_input("MLflow URI", value="http://localhost:5000")
st.sidebar.divider()
st.sidebar.caption("Refresh interval: 30s (cached)")

try:
    import mlflow

    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    models = list(client.search_registered_models())

    if models:
        for model in models:
            versions = client.get_latest_versions(model.name, stages=["Production", "Staging"])
            with st.expander(f"📦 {model.name}"):
                for v in versions:
                    color = "🟢" if v.current_stage == "Production" else "🟡"
                    st.write(f"{color} **{v.current_stage}** — v{v.version}")
                    st.caption(f"Run: `{v.run_id}` · Source: `{v.source}`")
    else:
        st.info("No registered models found.")

except Exception as exc:  # noqa: BLE001
    st.warning(f"MLflow unavailable: {exc}")


# ── Active A/B tests ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Active A/B Tests")

try:
    import asyncio

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from src.core.config import get_settings
    from src.core.models import ABTest

    async def _fetch_ab_tests() -> list[dict[str, object]]:
        settings = get_settings()
        engine = create_async_engine(settings.database_url)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            result = await session.execute(select(ABTest).where(ABTest.is_active == True))  # noqa: E712
            rows = result.scalars().all()
            return [
                {
                    "name": r.name,
                    "model_a": r.model_a,
                    "model_b": r.model_b,
                    "traffic_b_pct": r.traffic_b_pct,
                }
                for r in rows
            ]

    ab_tests = asyncio.run(_fetch_ab_tests())
    if ab_tests:
        for test in ab_tests:
            pct_b = float(test["traffic_b_pct"]) * 100
            label = (
                f"**{test['name']}**: {test['model_a']} ({100-pct_b:.0f}%)"
                f" vs {test['model_b']} ({pct_b:.0f}%)"
            )
            st.progress(pct_b / 100, text=label)
    else:
        st.info("No active A/B tests.")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Could not load A/B tests: {exc}")
