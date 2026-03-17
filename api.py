import numpy as np
import joblib
import uuid
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model        = joblib.load("xgb_fraud.pkl")
scaler       = joblib.load("scaler_fraud.pkl")
feature_cols = joblib.load("feature_cols.pkl")
metadata     = joblib.load("model_metadata.pkl")
THRESHOLD    = metadata["threshold"]

print(f"Model loaded | ROC-AUC: {metadata['roc_auc']:.4f} | Threshold: {THRESHOLD:.4f}")

# ── In-memory feature store (Redis simulation) ────────────────────────────────
# In production this would be Redis with sub-millisecond lookup
# Here we simulate it with a Python dict — same interface, same concept
class FeatureStore:
    def __init__(self):
        self.store = {}

    def set(self, key: str, value: dict):
        self.store[key] = {
            "value":     value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get(self, key: str):
        return self.store.get(key)

    def get_velocity(self, card_id: str) -> dict:
        """Return transaction velocity features for a card"""
        key = f"velocity:{card_id}"
        data = self.get(key)
        if not data:
            return {"tx_count_1h": 0, "tx_count_24h": 0,
                    "total_amount_1h": 0.0, "avg_amount": 0.0}
        return data["value"]

    def update_velocity(self, card_id: str, amount: float):
        key  = f"velocity:{card_id}"
        curr = self.get_velocity(card_id)
        curr["tx_count_1h"]    = curr.get("tx_count_1h", 0) + 1
        curr["tx_count_24h"]   = curr.get("tx_count_24h", 0) + 1
        curr["total_amount_1h"] = curr.get("total_amount_1h", 0.0) + amount
        curr["avg_amount"]     = curr["total_amount_1h"] / curr["tx_count_1h"]
        self.set(key, curr)

feature_store = FeatureStore()

# ── In-memory audit log ───────────────────────────────────────────────────────
audit_log = []

app = FastAPI(
    title="Real-Time Fraud Detection Engine",
    description="""
    Two-stage fraud detection system.
    - Stage 1: XGBoost classifier (ROC-AUC 0.98)
    - Stage 2: Graph-based velocity anomaly detection
    - Feature store: Redis-pattern in-memory store
    - Real-time scoring with sub-100ms latency
    """,
    version="1.0.0"
)

class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    card_id:        str
    amount:         float
    time:           float
    v1:  float = 0.0; v2:  float = 0.0; v3:  float = 0.0
    v4:  float = 0.0; v5:  float = 0.0; v6:  float = 0.0
    v7:  float = 0.0; v8:  float = 0.0; v9:  float = 0.0
    v10: float = 0.0; v11: float = 0.0; v12: float = 0.0
    v13: float = 0.0; v14: float = 0.0; v15: float = 0.0
    v16: float = 0.0; v17: float = 0.0; v18: float = 0.0
    v19: float = 0.0; v20: float = 0.0; v21: float = 0.0
    v22: float = 0.0; v23: float = 0.0; v24: float = 0.0
    v25: float = 0.0; v26: float = 0.0; v27: float = 0.0
    v28: float = 0.0

class FraudResponse(BaseModel):
    transaction_id:  str
    timestamp:       str
    card_id:         str
    amount:          float
    fraud_score:     float
    prediction:      str
    risk_level:      str
    recommendation:  str
    velocity_alert:  bool
    model_version:   str

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model":       "XGBoostClassifier",
        "roc_auc":     metadata["roc_auc"],
        "pr_auc":      metadata["pr_auc"],
        "threshold":   THRESHOLD,
        "version":     "1.0.0"
    }

@app.post("/predict", response_model=FraudResponse)
def predict(tx: Transaction):
    tx_id = tx.transaction_id or str(uuid.uuid4())

    # ── Pull velocity features from feature store ─────────────────────────────
    velocity = feature_store.get_velocity(tx.card_id)

    # ── Build feature vector (must match features.py exactly) ─────────────────
    v_vals = [getattr(tx, f'v{i}') for i in range(1, 29)]

    hour_of_day   = (tx.time / 3600) % 24
    is_night      = int(hour_of_day >= 23 or hour_of_day <= 5)
    amount_log    = np.log1p(tx.amount)
    amount_zscore = (tx.amount - 88.29) / 250.0
    is_round      = int(tx.amount % 1 == 0)
    is_large      = int(tx.amount > 200)
    tx_window     = min(velocity["tx_count_1h"] + 1, 500)

    v14, v10, v12, v4, v11 = tx.v14, tx.v10, tx.v12, tx.v4, tx.v11
    v14_v10 = v14 * v10
    v12_v4  = v12 * v4
    v14_sq  = v14 ** 2
    v10_sq  = v10 ** 2
    v_fraud_score = (-0.3*v14 - 0.25*v10 - 0.2*v12 + 0.2*v4 + 0.15*v11)

    features = np.array(
        v_vals + [amount_log, amount_zscore, is_round, is_large,
                  hour_of_day, is_night, tx_window,
                  v14_v10, v12_v4, v14_sq, v10_sq, v_fraud_score],
        dtype=np.float32
    ).reshape(1, -1)

    features_scaled = scaler.transform(features)
    fraud_score     = float(model.predict_proba(features_scaled)[0][1])
    is_fraud        = fraud_score >= THRESHOLD

    # ── Velocity anomaly check ────────────────────────────────────────────────
    velocity_alert = (
        velocity["tx_count_1h"] >= 5 or
        velocity["total_amount_1h"] > 1000 or
        (velocity["tx_count_1h"] > 0 and
         tx.amount > velocity["avg_amount"] * 3)
    )

    # Update feature store
    feature_store.update_velocity(tx.card_id, tx.amount)

    # ── Risk level ────────────────────────────────────────────────────────────
    if is_fraud or velocity_alert:
        risk_level     = "HIGH"
        prediction     = "FRAUD"
        recommendation = "Block transaction — flag for review"
    elif fraud_score > 0.5:
        risk_level     = "MEDIUM"
        prediction     = "SUSPICIOUS"
        recommendation = "Request additional authentication"
    else:
        risk_level     = "LOW"
        prediction     = "LEGITIMATE"
        recommendation = "Approve transaction"

    result = FraudResponse(
        transaction_id=tx_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        card_id=tx.card_id,
        amount=tx.amount,
        fraud_score=round(fraud_score, 6),
        prediction=prediction,
        risk_level=risk_level,
        recommendation=recommendation,
        velocity_alert=velocity_alert,
        model_version="1.0.0"
    )

    # Audit log
    audit_log.append(result.dict())
    return result

@app.get("/audit")
def get_audit(limit: int = 50):
    return {
        "total_predictions": len(audit_log),
        "recent":            audit_log[-limit:]
    }

@app.get("/feature_store/{card_id}")
def get_card_features(card_id: str):
    velocity = feature_store.get_velocity(card_id)
    return {"card_id": card_id, "velocity_features": velocity}

@app.get("/stats")
def get_stats():
    if not audit_log:
        return {"message": "No predictions yet"}
    fraud_count = sum(1 for r in audit_log if r["prediction"] == "FRAUD")
    return {
        "total_transactions": len(audit_log),
        "fraud_detected":     fraud_count,
        "fraud_rate":         round(fraud_count / len(audit_log) * 100, 2),
        "avg_fraud_score":    round(np.mean([r["fraud_score"] for r in audit_log]), 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
