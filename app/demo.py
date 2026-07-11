"""Streamlit demo UI for the fraud-detection API.

    make demo    (API must be running: make serve / make serve-shadow)

Not part of the production system - this is a human-friendly window onto it,
for demos and for explaining the project to non-technical audiences.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
from fraud_detection.drift import DRIFT_THRESHOLD, drift_report  # noqa: E402

RAW_DATA = Path("data/raw/creditcard.parquet")
REFERENCE = Path("models/reference_sample.parquet")
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🕵️", layout="wide")

# ---------- styling ----------

st.markdown(
    """
<style>
/* metric cards */
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { opacity: 0.75; }

/* hero banner */
.hero {
    background: linear-gradient(120deg, #0f2027 0%, #203a43 55%, #2c5364 100%);
    border-radius: 16px;
    padding: 28px 32px 24px;
    margin-bottom: 8px;
    color: white;
}
.hero h1 { color: white; font-size: 1.9rem; margin: 0 0 6px; }
.hero p  { color: #cfe3ee; margin: 0 0 12px; font-size: 1.02rem; }
.pill {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    color: #e8f1f7;
    border-radius: 999px;
    padding: 3px 12px;
    margin: 2px 6px 2px 0;
    font-size: 0.78rem;
    letter-spacing: 0.3px;
}

/* tabs a bit bolder */
button[data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }

/* primary button */
div.stButton > button[kind="primary"], div.stButton > button[data-testid="baseButton-primary"] {
    border-radius: 10px;
    font-weight: 700;
    padding: 0.6rem 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🕵️ Real-Time Fraud Detection</h1>
  <p>A production-grade ML system that scores card transactions in ~1 ms —
  and knows when its own answers can't be trusted.</p>
  <span class="pill">XGBoost</span><span class="pill">FastAPI</span>
  <span class="pill">MLflow</span><span class="pill">Docker</span>
  <span class="pill">Kubernetes</span><span class="pill">Terraform · AWS</span>
  <span class="pill">Prometheus · Grafana</span><span class="pill">PSI drift detection</span>
  <span class="pill">Shadow deployment</span>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- data helpers ----------

@st.cache_data
def load_split() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(legit, fraud) rows - full dataset locally, committed demo CSVs in the cloud."""
    if RAW_DATA.exists():
        df = pd.read_parquet(RAW_DATA)
    else:
        files = sorted(Path("samples").glob("demo_batch_*.csv"))
        if not files:
            return None, None
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df.rename(columns={"actual_fraud": "Class"}).drop(
            columns=["transaction_id"], errors="ignore"
        )
    return df[df["Class"] == 0], df[df["Class"] == 1]


@st.cache_data
def make_batch(seed: int, n_legit: int, n_fraud: int, high_value: bool = False) -> pd.DataFrame:
    """Build a sample 'bank batch file' with known ground truth mixed in."""
    legit, fraud = load_split()
    if high_value:
        legit = legit[legit["Amount"] > legit["Amount"].quantile(0.9)]
    batch = pd.concat(
        [legit.sample(n_legit, random_state=seed), fraud.sample(n_fraud, random_state=seed)]
    ).sample(frac=1.0, random_state=seed)  # shuffle so frauds aren't at the end
    batch = batch.rename(columns={"Class": "actual_fraud"}).reset_index(drop=True)
    batch.insert(0, "transaction_id", [f"TXN-{seed}{i:05d}" for i in range(len(batch))])
    return batch


def call_api(base_url: str, payload: dict) -> dict:
    r = httpx.post(f"{base_url}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def get_model_info(base_url: str) -> dict | None:
    try:
        r = httpx.get(f"{base_url}/model-info", timeout=5)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def scan(df: pd.DataFrame, base_url: str) -> pd.DataFrame:
    """Score every row through the live API with a progress bar."""
    progress = st.progress(0.0, text="Scanning transactions through the model...")
    probas = []
    with httpx.Client(base_url=base_url, timeout=10) as client:
        for i, row in enumerate(df[FEATURES].itertuples(index=False)):
            payload = dict(zip(FEATURES, (float(v) for v in row)))
            probas.append(client.post("/predict", json=payload).json()["fraud_probability"])
            progress.progress((i + 1) / len(df), text=f"Scanned {i + 1}/{len(df)}")
    progress.empty()
    out = df.copy()
    out["fraud_probability"] = probas
    out["verdict"] = ["🚨 BLOCK" if p >= 0.5 else "✅ approve" for p in probas]
    return out


RESULT_COLUMN_CONFIG = {
    "fraud_probability": st.column_config.ProgressColumn(
        "Fraud probability", format="percent", min_value=0.0, max_value=1.0
    ),
    "Amount": st.column_config.NumberColumn("Amount", format="£%.2f"),
    "transaction_id": st.column_config.TextColumn("Transaction"),
    "verdict": st.column_config.TextColumn("Verdict"),
    "actually_was": st.column_config.TextColumn("Actually was"),
}


# ---------- sidebar ----------

st.sidebar.title("🕵️ Fraud Detection")
base_url = st.sidebar.text_input(
    "API URL", os.environ.get("FRAUD_API_URL", "http://localhost:8000")
)

info = get_model_info(base_url)
if info:
    st.sidebar.success("API connected ✅")
    m = info["metrics"]
    s1, s2 = st.sidebar.columns(2)
    s1.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    s2.metric("Recall", f"{m['recall_at_threshold']:.0%}")
    st.sidebar.caption(f"Model run: `{info['run_id'][:12]}…`")
else:
    st.sidebar.error("API not reachable - run `make serve` first")

st.sidebar.divider()
st.sidebar.caption(
    "This UI is a demo window onto a production ML system: "
    "FastAPI model serving, Kubernetes deployment, Prometheus monitoring, "
    "drift detection, and automated champion/challenger retraining."
)

# ---------- main ----------

tab_batch, tab_score, tab_explain = st.tabs(
    ["📄  Scan a batch file", "🔍  Score one transaction", "📚  How it all works"]
)

# ----- batch scan -----
with tab_batch:
    st.subheader("Scan a file of transactions — like a bank would")
    st.write(
        "A payments team receives files with thousands of card transactions. "
        "Pick a sample file below (or upload your own), inspect it, then let "
        "the model scan it and flag the frauds hiding inside."
    )

    legit_df, _ = load_split()
    if legit_df is None:
        st.warning(
            "No data available - run `make train` (full dataset) or "
            "`python scripts/export_demo_assets.py` (demo samples), then reload."
        )
        st.stop()

    SAMPLES = {
        "Batch A - 200 everyday transactions (4 frauds hidden inside)": (1, 196, 4, False),
        "Batch B - 500 transactions, busy day (12 frauds hidden inside)": (2, 488, 12, False),
        "Batch C - 100 high-value transactions (3 frauds hidden inside)": (3, 97, 3, True),
        "Upload my own CSV": None,
    }
    choice = st.selectbox("Choose a transactions file", list(SAMPLES))

    batch = None
    if SAMPLES[choice] is None:
        up = st.file_uploader(
            "CSV with columns V1..V28 and Amount (optional: actual_fraud for scoring accuracy)",
            type="csv",
        )
        if up is not None:
            batch = pd.read_csv(up)
            # tolerate common label-column names from other datasets
            batch = batch.rename(columns={"Class": "actual_fraud", "class": "actual_fraud"})
            missing = set(FEATURES) - set(batch.columns)
            if missing:
                st.error(f"Missing columns: {sorted(missing)}")
                batch = None
            elif len(batch) > 2_000:
                st.info(
                    f"File has {len(batch):,} rows - scanning one HTTP call at a "
                    "time would take a while, so we scan a random sample."
                )
                n = st.slider("Rows to scan", 200, 5_000, 1_000, step=100)
                batch = batch.sample(n, random_state=42).reset_index(drop=True)
    else:
        seed, n_legit, n_fraud, high_value = SAMPLES[choice]
        batch = make_batch(seed, n_legit, n_fraud, high_value)

    if batch is not None:
        with st.expander(f"👀 Inspect the file — {len(batch)} transactions, exactly as the model receives them"):
            st.dataframe(batch.drop(columns=["actual_fraud"], errors="ignore"), height=240)
            st.download_button(
                "⬇ Download this file (CSV)",
                batch.to_csv(index=False),
                file_name="transactions_batch.csv",
            )
        st.caption(
            "Ground-truth labels are hidden from the model during the scan and "
            "only used afterwards to grade its answers."
        )

        if st.button("🔎 Scan this file for fraud", type="primary", use_container_width=True):
            try:
                results = scan(batch, base_url)
            except httpx.HTTPError as e:
                st.error(f"API call failed - is the server running? ({e})")
            else:
                flagged = results[results["fraud_probability"] >= 0.5]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Transactions scanned", f"{len(results):,}")
                c2.metric("Flagged as fraud", len(flagged))
                c3.metric("Money protected", f"£{flagged['Amount'].sum():,.0f}")

                if "actual_fraud" in results.columns:
                    truth = results["actual_fraud"] == 1
                    pred = results["fraud_probability"] >= 0.5
                    caught, total = int((truth & pred).sum()), int(truth.sum())
                    false_alarms = int((~truth & pred).sum())
                    c4.metric("Frauds caught", f"{caught} / {total}")
                    missed = total - caught
                    verdict_bits = [f"caught {caught} of {total} hidden frauds"]
                    if missed:
                        verdict_bits.append(f"missed {missed}")
                    verdict_bits.append(
                        f"{false_alarms} false alarm(s) on {int((~truth).sum())} genuine payments"
                    )
                    (st.success if missed == 0 else st.info)(
                        "The model " + ", ".join(verdict_bits) + "."
                    )

                # --- data health check ---
                if REFERENCE.exists():
                    ref = pd.read_parquet(REFERENCE)
                    dr = drift_report(ref, results[FEATURES])
                    n_drifted = int((dr["status"] == "drift").sum())
                    if n_drifted > 0:
                        worst = ", ".join(dr.head(5)["feature"])
                        st.warning(
                            f"⚠️ **Data health check: {n_drifted} of {len(dr)} features "
                            f"have drifted** beyond the PSI threshold ({DRIFT_THRESHOLD}) "
                            f"vs the training data (worst: {worst}). This file does not "
                            "look like what the model learned from - its predictions "
                            "here are unreliable, whatever they say. In production this "
                            "alarm would trigger the automated retraining pipeline."
                        )
                        with st.expander("Per-feature drift (PSI) details"):
                            st.dataframe(dr, height=300)
                    else:
                        st.success(
                            "✅ Data health check: no feature drift vs training data - "
                            "this file looks like what the model was trained on, so its "
                            "scores can be trusted."
                        )

                st.markdown("#### Scan results — most suspicious first")
                show = results.sort_values("fraud_probability", ascending=False)
                cols = ["transaction_id"] if "transaction_id" in show.columns else []
                cols += ["Amount", "fraud_probability", "verdict"]
                if "actual_fraud" in show.columns:
                    show["actually_was"] = show["actual_fraud"].map({1: "FRAUD", 0: "genuine"})
                    cols += ["actually_was"]
                st.dataframe(
                    show[cols],
                    column_config=RESULT_COLUMN_CONFIG,
                    height=420,
                    hide_index=True,
                )

# ----- single transaction -----
with tab_score:
    st.subheader("Is this transaction fraud?")
    st.caption(
        "Each button draws one random transaction from the real dataset. The "
        "model is NOT shown the true outcome - only the numbers. You compare "
        "its verdict against reality."
    )

    legit_df, fraud_df = load_split()
    col1, col2 = st.columns(2)
    sample, truth = None, None
    with col1:
        if st.button("🟢 Sample a REAL legitimate transaction", use_container_width=True):
            if legit_df is not None:
                sample, truth = legit_df.drop(columns="Class").sample(1), False
            else:
                st.warning("Run `make train` once to cache the dataset")
    with col2:
        if st.button("🔴 Sample a REAL fraudulent transaction", use_container_width=True):
            if fraud_df is not None:
                sample, truth = fraud_df.drop(columns="Class").sample(1), True
            else:
                st.warning("Run `make train` once to cache the dataset")

    if sample is not None:
        payload = {k: float(v) for k, v in sample.iloc[0].items()}
        try:
            result = call_api(base_url, payload)
        except httpx.HTTPError as e:
            st.error(f"API call failed: {e}")
        else:
            proba = result["fraud_probability"]
            st.progress(min(proba, 1.0))
            a, b, c = st.columns(3)
            a.metric("Ground truth (hidden from model)", "FRAUD" if truth else "LEGITIMATE")
            b.metric("Fraud probability", f"{proba:.1%}")
            c.metric("Model verdict", "🚨 BLOCK" if result["is_fraud"] else "✅ APPROVE")

            if result["is_fraud"] == truth:
                st.success(
                    "✔ Correct - the model "
                    + ("caught the fraud." if truth else "let the genuine payment through.")
                )
            else:
                st.error(
                    "✘ Wrong - the model "
                    + ("missed this fraud (a false negative). At ~84% recall, roughly "
                       "1 in 6 frauds slips through - this is one of them. No fraud "
                       "model is perfect; this is why thresholds and retraining matter."
                       if truth else
                       "blocked a genuine payment (a false positive) - the cost of "
                       "catching fraud aggressively.")
                )

            with st.expander("The 29 numbers the model saw (the actual input)"):
                st.write(
                    "V1-V28 are anonymised transforms of the original transaction "
                    "details (merchant, location, time patterns...) plus the amount. "
                    "This row was sent to the API exactly as you see it."
                )
                st.dataframe(sample.T.rename(columns=lambda _: "value"))

# ----- explainer -----
with tab_explain:
    st.subheader("The system behind this page")
    st.markdown(
        """
**The one-liner:** this project is the *full production life* of a machine-learning
model - not just training it, but serving, deploying, monitoring, and safely
replacing it. This page is the tip of the iceberg.

**1 — Training.** An XGBoost model learns from 284,807 real (anonymised) card
transactions, of which only 0.17% are fraud. Every experiment is tracked in
**MLflow** so results are reproducible and comparable.

**2 — Serving.** The trained model sits behind a **FastAPI** web service.
Anything that can make an HTTP request (this page, a payment system) can ask
it for a score. Typical answer time: ~1 ms.

**3 — Deployment.** The service is packaged with **Docker** and runs on
**Kubernetes**: 2+ copies for reliability, self-healing if one crashes,
auto-scaling under load. **Terraform** scripts can recreate the whole AWS
setup from scratch. **GitHub Actions** tests and ships every code change.

**4 — Monitoring.** **Prometheus** collects live metrics (traffic, latency,
fraud rate) and **Grafana** charts them. Every prediction is logged.

**5 — Drift detection.** Fraud patterns change. The system compares live
traffic against the training data using **PSI** (the metric banks use) and
raises an alarm when the world has shifted from what the model learned.
You can see this live: upload a foreign dataset in the scan tab and watch
the data health check fire.

**6 — Safe retraining.** On drift, a *challenger* model is trained. It shadows
production - scoring real traffic silently while the *champion* still makes
all decisions - and is **promoted only if measurably better**. This repo's
model v2 replaced v1 exactly this way (PR-AUC 0.8817 > 0.8799).
        """
    )
    st.info(
        "Analogy: the model is a sniffer dog at an airport. Training school "
        "(MLflow) certifies it; the airport post (Kubernetes) keeps it working "
        "in shifts with backup dogs; supervisors (Grafana) watch its accuracy; "
        "and when smugglers change tactics (drift), a new dog is trained and "
        "walks alongside the old one (shadow) before taking over (promotion)."
    )
