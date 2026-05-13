# 🛡️ Real-Time Fraud Detection Engine

A two-stage credit card fraud detection system built on 284,807 real transactions with a 1:577 class imbalance. XGBoost classifier with 40 engineered features achieves ROC-AUC 0.9819 and PR-AUC 0.7587 — optimized for precision-first detection on highly skewed data.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Live Dashboard

👉 **[View Live App](https://gdiaz38-fraud-detection-engine.streamlit.app)**

---

## Overview

Credit card fraud costs the global economy over $32B annually. This project builds a real-time fraud scoring engine on the ULB Credit Card dataset — one of the most imbalanced real-world classification problems available, with only 492 fraud cases in 284,807 transactions (0.17%).

Key question it answers: *Is this transaction fraudulent — and how confident are we?*

---

## Key Results

| Metric | Value |
|---|---|
| ROC-AUC | **0.9819** |
| PR-AUC | **0.7587** |
| Fraud cases caught | 91 of 98 test cases |
| Class imbalance | 1:577 (fraud:legitimate) |
| Features | 40 engineered |
| Threshold | 0.9996 (F1-optimized on PR curve) |

---

## Features

- **Real XGBoost inference** — trained model scores every transaction on load
- **Live transaction scorer** — pick any real test transaction (legitimate or fraud), score it through the actual model
- **Full score distribution** — histogram showing separation between fraud and legitimate scores
- **Business impact analysis** — dollars saved vs missed vs false alarm cost
- **Real-time stream** — streams real test transactions live with score chart updating per transaction
- **Audit log** — every scored transaction logged with UUID, timestamp, score, true label, card velocity

---

## Data

| Source | Description |
|---|---|
| [ULB Credit Card Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 real European cardholder transactions (Sept 2013) |
| Features V1–V28 | PCA-transformed (anonymized for privacy) |
| Features Time, Amount | Raw — time since first transaction, transaction amount |
| Class imbalance | 492 fraud / 284,315 legitimate (0.172%) |

---

## Project Structure

```
fraud-detection-engine/
├── dashboard.py              # Streamlit — 4 tabs: scoring, performance, stream, audit
├── train.py                  # XGBoost training with scale_pos_weight + PR-AUC optimization
├── features.py               # Feature engineering — V interactions, velocity, time signals
├── api.py                    # FastAPI REST endpoint with in-memory feature store
├── graph_detection.py        # NetworkX transaction graph layer
├── xgb_fraud.pkl             # Trained XGBoost model
├── scaler_fraud.pkl          # StandardScaler (fitted on training set)
├── feature_cols.pkl          # Feature column names (40 features)
├── model_metadata.pkl        # ROC-AUC, PR-AUC, optimal threshold
├── X_test.npy                # Test features (56,962 transactions)
├── X_train.npy               # Training features
├── y_test.npy                # Test labels
├── y_train.npy               # Training labels
└── requirements.txt
```

---

## How It Works

```
Raw transaction (Time, Amount, V1-V28)
        ↓
features.py engineers 40 features:
  V-feature interaction terms (V1×V3, V4×V11, etc.)
  Time signals (hour of day, is_night, is_weekend)
  Velocity metrics (rolling amount, frequency)
  Amount statistics (log_amount, amount_zscore)
        ↓
XGBoost classifier with scale_pos_weight=577
Trained on balanced sample weights
Threshold = F1-maximizing point on precision-recall curve
        ↓
Fraud probability scored in real time
Classified: FRAUD | SUSPICIOUS | LEGITIMATE
        ↓
Every prediction logged to audit trail
```

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Trees | 500 (early stopping 30 rounds) |
| Max depth | 6 |
| Learning rate | 0.05 |
| `scale_pos_weight` | 577 (fraud upweight) |
| Eval metric | PR-AUC (better than ROC for imbalanced) |
| Threshold | 0.9996 (F1-optimal on PR curve) |

**Why such a high threshold?** With 0.17% fraud rate, a standard 0.5 threshold produces thousands of false positives. Optimizing on the precision-recall curve sets a threshold that maximizes F1, resulting in 0.9996 — effectively requiring near-certainty before flagging.

---

## Business Impact (Test Set)

| Outcome | Count | Financial Impact |
|---|---|---|
| True Positives (caught) | 91 | +$11,131 saved |
| False Negatives (missed) | 7 | -$855 lost |
| False Positives (false alarms) | ~50 | -$125 review cost |
| **Net savings** | | **~$10,151** |

At $122.21 avg fraud amount and $2.50 per false alarm investigation.

---

## Dashboard Tabs

**Live Scoring** — pick a real test transaction (random legitimate, random fraud, or by index), view feature values, score through model, see fraud gauge and velocity alert

**Model Performance** — score distribution histogram, confusion matrix, business impact metrics, top 15 feature importances

**Transaction Stream** — stream up to 200 real test transactions in real time, watch live fraud score chart update, track detection rate and false positives as they accumulate

**Audit Log** — full history of scored transactions with prediction coloring, downloadable as CSV

---

## Local Setup

```bash
git clone https://github.com/gdiaz38/fraud-detection-engine
cd fraud-detection-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

To retrain from scratch (requires ULB dataset via kagglehub):

```bash
python3 features.py   # engineer features → .npy files
python3 train.py      # train XGBoost → xgb_fraud.pkl
```

---

## Tech Stack

`Python 3.11` · `XGBoost` · `Streamlit` · `Plotly` · `Pandas` · `NumPy` · `Scikit-learn` · `joblib`

---

## Affiliation

University of California, Riverside — MS in Engineering Management
Part of a portfolio of 10 live data science projects spanning computer vision, NLP, supply chain, and healthcare ML.

---

## License

MIT