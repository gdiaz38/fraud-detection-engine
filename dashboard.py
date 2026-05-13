import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, os, uuid, time
from datetime import datetime

st.set_page_config(page_title="Fraud Detection Engine", page_icon="🛡️", layout="wide")
st.markdown('<h1 style="color:#ff4444">🛡️ Real-Time Fraud Detection Engine</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#888">XGBoost · ROC-AUC 0.9819 · PR-AUC 0.7587 · 40 engineered features · 1:577 class imbalance</p>', unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model  = joblib.load(os.path.join(BASE, "xgb_fraud.pkl"))
    meta   = joblib.load(os.path.join(BASE, "model_metadata.pkl"))
    fc     = joblib.load(os.path.join(BASE, "feature_cols.pkl"))
    return model, meta, fc

@st.cache_data
def load_data():
    X_test = np.load(os.path.join(BASE, "X_test.npy"))
    y_test = np.load(os.path.join(BASE, "y_test.npy"))
    return X_test, y_test

@st.cache_data
def get_all_scores():
    X_test, y_test = load_data()
    model, meta, _ = load_model()
    # X_test is NOT pre-scaled — pass directly to model
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= meta["threshold"]).astype(int)
    return y_prob, y_pred

model, meta, feature_cols = load_model()
X_test, y_test             = load_data()
y_prob, y_pred             = get_all_scores()
THRESHOLD                  = meta["threshold"]

fraud_idx  = np.where(y_test == 1)[0]
legit_idx  = np.where(y_test == 0)[0]

tp = int(((y_pred == 1) & (y_test == 1)).sum())
fp = int(((y_pred == 1) & (y_test == 0)).sum())
fn = int(((y_pred == 0) & (y_test == 1)).sum())
tn = int(((y_pred == 0) & (y_test == 0)).sum())

def classify(score):
    if score >= THRESHOLD:
        return "FRAUD",      "#ff4444", "🔴", "Block immediately. Flag card for review."
    if score >= THRESHOLD * 0.85:
        return "SUSPICIOUS", "#ffa500", "🟠", "Hold for manual review. Send OTP."
    return     "LEGITIMATE", "#00d4aa", "🟢", "Approve transaction."

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "velocity" not in st.session_state:
    st.session_state.velocity = {}

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Test Transactions", f"{len(y_test):,}")
k2.metric("True Fraud Cases",  f"{y_test.sum()}")
k3.metric("ROC-AUC",           f"{meta['roc_auc']:.4f}")
k4.metric("PR-AUC",            f"{meta['pr_auc']:.4f}")
k5.metric("Fraud Caught",      f"{tp}/{y_test.sum()}",
          f"{tp/y_test.sum():.0%} recall")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Scoring", "📊 Model Performance",
    "🔄 Transaction Stream", "🗂️ Audit Log"
])

# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Score a Real Test Transaction")

    col1, col2 = st.columns([1, 2])
    with col1:
        card_id   = st.text_input("Card ID", value="CARD-001")
        tx_type   = st.radio("Pick transaction type",
                             ["Random Legitimate", "Random Fraud", "By Index"])

        if tx_type == "Random Legitimate":
            idx = int(np.random.choice(legit_idx))
        elif tx_type == "Random Fraud":
            idx = int(np.random.choice(fraud_idx))
        else:
            idx = st.number_input("Test set index", 0, len(X_test)-1, 0)

        tx      = X_test[idx]
        true_label = "FRAUD" if y_test[idx] == 1 else "LEGITIMATE"
        amount  = abs(float(tx[-1])) * 100  # Amount is last feature
        st.caption(f"Transaction #{idx} | True label: **{true_label}**")

    with col2:
        top_features = feature_cols[:10]
        fig_feat = go.Figure(go.Bar(
            x=[float(tx[feature_cols.index(f)]) for f in top_features],
            y=top_features, orientation="h",
            marker_color=["#ff4444" if v < -2 else "#00d4aa"
                          for v in [float(tx[feature_cols.index(f)])
                                    for f in top_features]]
        ))
        fig_feat.update_layout(template="plotly_dark", height=280,
                               title="Top Feature Values",
                               xaxis_title="Value", yaxis_title="")
        st.plotly_chart(fig_feat, use_container_width=True)

    if st.button("🔍 Score Transaction", type="primary"):
        score  = float(y_prob[idx])
        pred, color, icon, rec = classify(score)

        vc = st.session_state.velocity
        vc[card_id] = vc.get(card_id, 0) + 1
        vel_alert   = vc[card_id] >= 3

        st.session_state.audit_log.append({
            "record_id":      str(uuid.uuid4())[:16],
            "card_id":        card_id,
            "tx_index":       idx,
            "amount":         round(amount, 2),
            "fraud_score":    round(score, 6),
            "prediction":     pred,
            "true_label":     true_label,
            "correct":        pred == true_label,
            "velocity":       vc[card_id],
            "timestamp":      datetime.utcnow().isoformat()[:19]
        })

        st.markdown(f"""
        <div style="background:#1a1a1a;border:2px solid {color};
                    border-radius:10px;padding:1.5rem;text-align:center">
            <h2 style="color:{color}">{icon} {pred}</h2>
            <h1 style="color:{color}">Score: {score:.6f}</h1>
            <p style="color:#aaa">{rec}</p>
            <p style="color:#666">Threshold: {THRESHOLD:.6f} | True label: {true_label}</p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fraud Score",    f"{score:.6f}")
        m2.metric("Threshold",      f"{THRESHOLD:.4f}")
        m3.metric("Velocity Alert", "⚠️ YES" if vel_alert else "✅ NO")
        m4.metric("Card Txs",       f"{vc[card_id]} this session")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "valueformat": ".4f"},
            title={"text": "Fraud Probability (%)"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": color},
                "steps": [
                    {"range": [0,  THRESHOLD*100*0.85], "color": "#0d2b1f"},
                    {"range": [THRESHOLD*100*0.85, THRESHOLD*100], "color": "#2b1f0d"},
                    {"range": [THRESHOLD*100, 100], "color": "#2b0d0d"}
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "value": THRESHOLD * 100}
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance — Full Test Set")

    col1, col2 = st.columns(2)

    with col1:
        # Score distribution
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=y_prob[y_test == 0], nbinsx=80,
            name="Legitimate", marker_color="#00d4aa", opacity=0.7
        ))
        fig1.add_trace(go.Histogram(
            x=y_prob[y_test == 1], nbinsx=40,
            name="Fraud", marker_color="#ff4444", opacity=0.9
        ))
        fig1.add_vline(x=THRESHOLD, line_dash="dash", line_color="white",
                       annotation_text=f"Threshold={THRESHOLD:.4f}")
        fig1.update_layout(template="plotly_dark", height=340,
                           title="Fraud Score Distribution",
                           xaxis_title="Fraud Probability",
                           barmode="overlay", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        fig2 = go.Figure(go.Heatmap(
            z=cm, text=cm, texttemplate="%{text}",
            x=["Pred: Legit", "Pred: Fraud"],
            y=["True: Legit", "True: Fraud"],
            colorscale="Blues", showscale=False
        ))
        fig2.update_layout(template="plotly_dark", height=340,
                           title="Confusion Matrix")
        st.plotly_chart(fig2, use_container_width=True)

    # Business impact
    st.subheader("💰 Business Impact Analysis")
    avg_fraud = 122.21
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Fraud Caught (TP)",   tp, f"${tp*avg_fraud:,.0f} saved")
    b2.metric("Fraud Missed (FN)",   fn, f"-${fn*avg_fraud:,.0f} lost")
    b3.metric("False Alarms (FP)",   fp, f"-${fp*2.50:,.0f} review cost")
    b4.metric("Net Savings",
              f"${tp*avg_fraud - fn*avg_fraud - fp*2.50:,.0f}")

    # Feature importance
    st.subheader("Top 15 Feature Importances")
    imp     = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]
    fig3 = go.Figure(go.Bar(
        x=imp[top_idx],
        y=[feature_cols[i] for i in top_idx],
        orientation="h", marker_color="#ff4444"
    ))
    fig3.update_layout(template="plotly_dark", height=420,
                       xaxis_title="Importance",
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Real-Time Transaction Stream — Live Scoring")
    st.caption("Streams real test transactions one at a time through the model")

    n_stream   = st.slider("Transactions to stream", 20, 200, 50)
    include_fraud = st.checkbox("Ensure fraud cases are included", value=True)
    speed      = st.slider("Speed (tx/sec)", 1, 20, 8)

    if st.button("▶ Start Stream", type="primary"):
        # Mix real fraud and legit transactions
        if include_fraud:
            n_fraud = max(5, int(n_stream * 0.15))
            n_legit = n_stream - n_fraud
            stream_fraud = np.random.choice(fraud_idx,
                           min(n_fraud, len(fraud_idx)), replace=False)
            stream_legit = np.random.choice(legit_idx, n_legit, replace=False)
            stream_indices = np.concatenate([stream_fraud, stream_legit])
        else:
            stream_indices = np.random.choice(len(X_test), n_stream, replace=False)

        np.random.shuffle(stream_indices)

        placeholder = st.empty()
        history = []

        for i, idx in enumerate(stream_indices):
            score      = float(y_prob[idx])
            true_lbl   = int(y_test[idx])
            pred, color, icon, _ = classify(score)
            correct    = (pred == "FRAUD") == (true_lbl == 1)
            history.append({
                "tx": i+1, "score": score,
                "pred": pred, "true": true_lbl, "correct": correct
            })

            with placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Streamed",    f"{i+1}/{n_stream}")
                c2.metric("Fraud Caught",
                          sum(1 for h in history
                              if h["pred"]=="FRAUD" and h["true"]==1))
                c3.metric("False Positives",
                          sum(1 for h in history
                              if h["pred"]=="FRAUD" and h["true"]==0))
                c4.metric("Accuracy",
                          f"{sum(h['correct'] for h in history)/len(history):.1%}")

                fig4 = go.Figure()
                h_df = pd.DataFrame(history)
                fig4.add_trace(go.Scatter(
                    x=h_df["tx"], y=h_df["score"],
                    mode="lines+markers",
                    marker=dict(
                        color=["#ff4444" if p=="FRAUD"
                               else "#ffa500" if p=="SUSPICIOUS"
                               else "#00d4aa" for p in h_df["pred"]],
                        size=8
                    ),
                    line=dict(color="#444", width=1)
                ))
                fig4.add_hline(y=THRESHOLD, line_dash="dash",
                               line_color="white",
                               annotation_text="Fraud threshold")
                fig4.update_layout(
                    template="plotly_dark", height=300,
                    title="Live Fraud Score Stream",
                    yaxis_title="Fraud Score", xaxis_title="Transaction"
                )
                st.plotly_chart(fig4, use_container_width=True)

            time.sleep(1.0 / speed)

        st.success(f"✅ Streamed {n_stream} transactions")

# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🗂️ Audit Log")
    log = st.session_state.audit_log
    if log:
        df_audit = pd.DataFrame(log)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scored",    len(df_audit))
        c2.metric("Flagged Fraud",
                  len(df_audit[df_audit["prediction"]=="FRAUD"]))
        c3.metric("Correct",
                  f"{df_audit['correct'].mean():.1%}" if "correct" in df_audit else "—")
        c4.metric("Unique Cards",    df_audit["card_id"].nunique())

        def color_pred(val):
            if val == "FRAUD":      return "color:#ff4444"
            if val == "SUSPICIOUS": return "color:#ffa500"
            return "color:#00d4aa"

        st.dataframe(
            df_audit.style.applymap(color_pred, subset=["prediction"]),
            use_container_width=True
        )
        st.download_button("⬇ Download Audit CSV",
                           df_audit.to_csv(index=False),
                           "fraud_audit.csv", "text/csv")
    else:
        st.info("No transactions scored yet.")

    st.markdown("---")
    st.markdown("""
    ### System Architecture
    - **XGBoost Classifier** — 40 engineered features, scale_pos_weight=577
    - **Threshold** — optimized on PR curve (F1-maximizing)
    - **Velocity detection** — per-card transaction frequency monitoring
    - **Feature store** — Redis-pattern in-memory store
    - **Audit trail** — every prediction logged with UUID and timestamp
    """)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Stats")
    st.markdown(f"**Test samples:** {len(y_test):,}")
    st.markdown(f"**True fraud:** {int(y_test.sum())} ({y_test.mean():.2%})")
    st.markdown(f"**ROC-AUC:** {meta['roc_auc']:.4f}")
    st.markdown(f"**PR-AUC:** {meta['pr_auc']:.4f}")
    st.markdown(f"**Threshold:** {THRESHOLD:.4f}")
    st.markdown(f"**Recall:** {tp}/{int(y_test.sum())}")
    st.markdown(f"**Precision:** {tp}/{tp+fp if tp+fp>0 else 1:.0f}")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("- ULB Credit Card (Kaggle)")
    st.markdown("- 284,807 transactions")
    st.markdown("- 1:577 fraud ratio")
    st.markdown("---")
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()
