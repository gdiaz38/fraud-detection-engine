import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random

st.set_page_config(
    page_title="Fraud Detection Engine",
    page_icon="🛡️",
    layout="wide"
)

API_URL = "http://localhost:8003"

st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:700; color:#ff4444; }
    .sub-header  { color:#888; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Real-Time Fraud Detection Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">XGBoost classifier (ROC-AUC 0.98) + velocity anomaly detection + in-memory feature store | ULB Credit Card Dataset</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["⚡ Live Scoring", "📊 Analytics", "🗂️ Audit Log"])

# ── Fraud V-feature profiles ──────────────────────────────────────────────────
LEGIT_PROFILE  = {"v14": -0.1, "v10": 0.2,  "v12": 0.1,  "v4": 0.3,  "v11": 0.1}
FRAUD_PROFILE  = {"v14": -5.2, "v10": -4.1, "v12": -3.8, "v4": 4.5,  "v11": 3.2}
MEDIUM_PROFILE = {"v14": -1.8, "v10": -1.5, "v12": -1.2, "v4": 1.8,  "v11": 1.2}

def make_transaction(card_id, amount, time_val, profile, noise=0.1):
    tx = {"card_id": card_id, "amount": amount, "time": time_val}
    for k, v in profile.items():
        tx[k] = round(v + random.gauss(0, noise), 4)
    return tx

def score_transaction(tx):
    try:
        r = requests.post(f"{API_URL}/predict", json=tx, timeout=5)
        return r.json()
    except:
        return None

# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Single Transaction Scorer")

    col1, col2 = st.columns(2)
    with col1:
        card_id    = st.text_input("Card ID", value="CARD-TEST-001")
        amount     = st.number_input("Amount ($)", min_value=0.01,
                                     max_value=10000.0, value=99.99)
        time_val   = st.slider("Time (hours since midnight)", 0, 47, 12) * 3600
        profile    = st.selectbox("Transaction profile",
                                  ["Legitimate", "Suspicious", "Fraud"])

    with col2:
        v14 = st.slider("V14 (fraud indicator)", -10.0, 5.0,
                        LEGIT_PROFILE["v14"] if profile=="Legitimate"
                        else FRAUD_PROFILE["v14"] if profile=="Fraud"
                        else MEDIUM_PROFILE["v14"])
        v10 = st.slider("V10", -10.0, 5.0,
                        LEGIT_PROFILE["v10"] if profile=="Legitimate"
                        else FRAUD_PROFILE["v10"] if profile=="Fraud"
                        else MEDIUM_PROFILE["v10"])
        v12 = st.slider("V12", -10.0, 5.0,
                        LEGIT_PROFILE["v12"] if profile=="Legitimate"
                        else FRAUD_PROFILE["v12"] if profile=="Fraud"
                        else MEDIUM_PROFILE["v12"])

    if st.button("🔍 Score Transaction", type="primary"):
        tx = {"card_id": card_id, "amount": amount, "time": time_val,
              "v14": v14, "v10": v10, "v12": v12}
        with st.spinner("Scoring..."):
            result = score_transaction(tx)

        if result:
            pred  = result["prediction"]
            score = result["fraud_score"]
            color = "#ff4444" if pred=="FRAUD" else \
                    "#ffa500" if pred=="SUSPICIOUS" else "#00d4aa"

            st.markdown(f"""
            <div style="background:#1a1a1a;border:2px solid {color};
                        border-radius:10px;padding:1.5rem;text-align:center;
                        margin:1rem 0">
                <h2 style="color:{color};margin:0">{pred}</h2>
                <h1 style="color:{color};margin:0.5rem 0">
                    Fraud Score: {score:.4f}
                </h1>
                <p style="color:#aaa">{result['recommendation']}</p>
                <p style="color:#666;font-size:0.8rem">
                    TX: {result['transaction_id'][:16]}... |
                    {result['timestamp'][:19]}
                </p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Level",     result["risk_level"])
            c2.metric("Velocity Alert", "⚠️ YES" if result["velocity_alert"] else "✅ NO")
            c3.metric("Amount",         f"${result['amount']:.2f}")

            # Fraud score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                title={"text": "Fraud Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  50], "color": "#0d2b1f"},
                        {"range": [50, 75], "color": "#2b1f0d"},
                        {"range": [75, 100],"color": "#2b0d0d"}
                    ],
                    "threshold": {
                        "line":  {"color": "white", "width": 3},
                        "value": 99.96
                    }
                }
            ))
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Transaction Stream Simulation")
    n_transactions = st.slider("Transactions to simulate", 20, 100, 50)
    fraud_rate     = st.slider("Injected fraud rate (%)", 5, 40, 15)

    if st.button("▶ Run Simulation", type="primary"):
        results  = []
        progress = st.progress(0)
        status   = st.empty()

        for i in range(n_transactions):
            is_fraud = random.random() < (fraud_rate / 100)
            profile  = FRAUD_PROFILE if is_fraud else LEGIT_PROFILE
            noise    = 0.3 if is_fraud else 0.2
            amount   = random.uniform(500, 2000) if is_fraud \
                       else random.uniform(5, 300)
            card_id  = f"CARD-{random.randint(1,20):03d}"
            tx       = make_transaction(card_id, amount,
                                        random.uniform(0, 86400),
                                        profile, noise)
            result   = score_transaction(tx)
            if result:
                results.append({
                    "tx":          i+1,
                    "card_id":     card_id,
                    "amount":      amount,
                    "fraud_score": result["fraud_score"],
                    "prediction":  result["prediction"],
                    "injected":    "FRAUD" if is_fraud else "LEGIT"
                })
            progress.progress((i+1)/n_transactions)
            status.text(f"Scored {i+1}/{n_transactions} transactions...")

        df = pd.DataFrame(results)
        status.empty()

        # Metrics
        detected = len(df[(df["prediction"]=="FRAUD") & (df["injected"]=="FRAUD")])
        total_fraud = len(df[df["injected"]=="FRAUD"])
        false_pos   = len(df[(df["prediction"]=="FRAUD") & (df["injected"]=="LEGIT")])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scored",    n_transactions)
        c2.metric("Fraud Detected",  f"{detected}/{total_fraud}")
        c3.metric("False Positives", false_pos)
        c4.metric("Avg Fraud Score",
                  f"{df[df['injected']=='FRAUD']['fraud_score'].mean():.3f}")

        # Score distribution
        fig = make_subplots(1, 2, subplot_titles=(
            "Fraud Score Distribution", "Predictions vs Injected"))

        fig.add_trace(go.Histogram(
            x=df[df["injected"]=="LEGIT"]["fraud_score"],
            name="Legitimate", marker_color="#00d4aa", opacity=0.7
        ), row=1, col=1)
        fig.add_trace(go.Histogram(
            x=df[df["injected"]=="FRAUD"]["fraud_score"],
            name="Fraud", marker_color="#ff4444", opacity=0.7
        ), row=1, col=1)

        pred_counts = df["prediction"].value_counts()
        fig.add_trace(go.Bar(
            x=pred_counts.index,
            y=pred_counts.values,
            marker_color=["#ff4444" if x=="FRAUD"
                          else "#ffa500" if x=="SUSPICIOUS"
                          else "#00d4aa"
                          for x in pred_counts.index]
        ), row=1, col=2)

        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Transaction log"):
            st.dataframe(df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Audit Log & System Stats")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Get System Stats"):
            try:
                r = requests.get(f"{API_URL}/stats", timeout=5)
                stats = r.json()
                if "total_transactions" in stats:
                    st.metric("Total Transactions", stats["total_transactions"])
                    st.metric("Fraud Detected",     stats["fraud_detected"])
                    st.metric("Fraud Rate",          f"{stats['fraud_rate']}%")
                    st.metric("Avg Fraud Score",     stats["avg_fraud_score"])
                else:
                    st.info(stats["message"])
            except Exception as e:
                st.error(f"API error: {e}")

    with col2:
        if st.button("📋 View Recent Predictions"):
            try:
                r    = requests.get(f"{API_URL}/audit?limit=20", timeout=5)
                data = r.json()
                if data["recent"]:
                    df_audit = pd.DataFrame(data["recent"])
                    st.dataframe(
                        df_audit[["transaction_id","card_id","amount",
                                  "fraud_score","prediction","timestamp"]],
                        use_container_width=True
                    )
                    st.download_button(
                        "⬇ Download Audit CSV",
                        df_audit.to_csv(index=False),
                        "fraud_audit.csv", "text/csv"
                    )
            except Exception as e:
                st.error(f"API error: {e}")

    st.markdown("---")
    st.markdown("""
    ### System Architecture
    - **XGBoost Classifier** — 40 engineered features, ROC-AUC 0.9819
    - **Velocity Detection** — per-card transaction frequency and amount monitoring
    - **Feature Store** — Redis-pattern in-memory store for sub-millisecond feature lookup
    - **Graph Layer** — NetworkX transaction graph (merchant/time window nodes)
    - **Audit Trail** — every prediction logged with UUID, timestamp, score, card ID
    """)
