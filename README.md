# 🛡️ Real-Time Fraud Detection Engine

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge)](https://fraud-detection-engine-gdiaz38.streamlit.app/)

Production-style fraud detection system built on 284k real European credit card 
transactions. Two-stage detection: XGBoost classifier + card velocity anomaly detection, 
deployed as a REST API with a live monitoring dashboard.

---

## 🚀 Try It Live

**[→ Open Live Dashboard](https://fraud-detection-engine-gdiaz38.streamlit.app/)**

Score individual transactions, run batch simulations with adjustable fraud rates, 
and monitor the real-time audit log.

---
## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9819 |
| PR-AUC | 0.7587 |
| Recall (fraud) | 74% |
| Precision (fraud) | 80% |
| Business impact | $8,921 saved vs $3,055 missed per test batch |

---

## Architecture
```
284k real transactions (ULB Credit Card Dataset)
            ↓
Feature Engineering (40 features)
- V1-V28 PCA features
- Amount log transform + z-score
- Time-based features (hour, is_night)
- V-feature interactions (V14×V10, V12×V4)
- Fraud signal composite score
            ↓
XGBoost Classifier
scale_pos_weight=577x (handles 1:577 class imbalance)
            ↓
Velocity Anomaly Layer
- Per-card transaction count (1h window)
- Per-card spend velocity
- Spike detection (3x avg amount)
            ↓
In-Memory Feature Store (Redis pattern)
            ↓
FastAPI REST endpoint (/predict)
            ↓
Streamlit real-time dashboard
```

---

## Key Design Decisions

**Class imbalance** — 1 in 577 transactions is fraud. 
Used `scale_pos_weight=577` and optimized threshold on 
precision-recall curve (not accuracy) — standard practice 
in production fraud systems.

**PR-AUC over ROC-AUC** — ROC-AUC is misleading on 
imbalanced datasets. PR-AUC of 0.76 on this dataset 
is the metric that actually matters.

**Feature store pattern** — velocity features are 
pre-computed and cached per card ID, enabling 
sub-millisecond lookup at inference time. 
Production systems use Redis for this.

**Graph layer** — transaction graph built with NetworkX 
connecting time windows and amount clusters. 
On real data with merchant IDs and card IDs, 
graph centrality features add 3-5% AUC lift.

---

## Stack

- **Model:** XGBoost (scikit-learn API)
- **Graph:** NetworkX
- **Feature store:** In-memory dict (Redis pattern)
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit + Plotly

---

## Run It
```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn xgboost fastapi uvicorn streamlit plotly joblib kagglehub networkx

python3 download_data.py
python3 features.py
python3 train.py
python3 graph_detection.py

# Terminal 1
python3 api.py

# Terminal 2
streamlit run dashboard.py
```

## API
```bash
POST /predict
{
  "card_id": "CARD-001",
  "amount": 1200.00,
  "time": 43200,
  "v14": -5.2,
  "v10": -4.1
}

GET /stats
GET /audit?limit=50
GET /feature_store/{card_id}
```
