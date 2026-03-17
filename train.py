import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve)
from sklearn.utils.class_weight import compute_sample_weight

print("Loading data...")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

print(f"Train fraud rate: {y_train.mean()*100:.3f}%")
print(f"Test  fraud rate: {y_test.mean()*100:.3f}%")

# ── Handle class imbalance with sample weights ────────────────────────────────
# This tells XGBoost to penalize missing fraud cases much more than
# missing legitimate ones — critical for real fraud detection
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
fraud_weight   = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"Fraud upweight factor: {fraud_weight:.1f}x")

# ── XGBoost Classifier ────────────────────────────────────────────────────────
print("\nTraining XGBoost fraud detector...")
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=fraud_weight,  # key param for imbalanced data
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric='aucpr'             # area under PR curve — better than AUC for imbalanced
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    verbose=50
)

joblib.dump(model, "xgb_fraud.pkl")
print("✅ Saved xgb_fraud.pkl")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]

# Find optimal threshold using F1 on precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores  = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx   = np.argmax(f1_scores)
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
y_pred     = (y_prob >= best_thresh).astype(int)

print(f"\nOptimal threshold: {best_thresh:.4f}")
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
print(f"ROC-AUC:           {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC:            {average_precision_score(y_test, y_prob):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraud'])}")

# Cost analysis — real business impact
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
avg_fraud_amount = 122.21
false_neg_cost   = fn * avg_fraud_amount
false_pos_cost   = fp * 2.50  # cost of investigating a false alarm
print(f"=== BUSINESS IMPACT ===")
print(f"True Positives  (fraud caught):    {tp:>4} — ${tp*avg_fraud_amount:>10,.2f} saved")
print(f"False Negatives (fraud missed):    {fn:>4} — ${false_neg_cost:>10,.2f} lost")
print(f"False Positives (wrong flags):     {fp:>4} — ${false_pos_cost:>10,.2f} investigation cost")
print(f"True Negatives  (correct clears):  {tn:>4}")

# Save threshold and metadata
joblib.dump({
    "threshold": float(best_thresh),
    "roc_auc":   float(roc_auc_score(y_test, y_prob)),
    "pr_auc":    float(average_precision_score(y_test, y_prob))
}, "model_metadata.pkl")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Precision-Recall curve
axes[0].plot(recalls, precisions, color='#00d4aa', linewidth=2)
axes[0].scatter(recalls[best_idx], precisions[best_idx],
                color='red', s=100, zorder=5, label=f'Best F1 @ {best_thresh:.2f}')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title(f'Precision-Recall Curve\nPR-AUC={average_precision_score(y_test, y_prob):.4f}')
axes[0].legend()

# 2. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Legitimate','Fraud'],
            yticklabels=['Legitimate','Fraud'])
axes[1].set_title('Confusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

# 3. Feature importance
feature_cols = joblib.load("feature_cols.pkl")
importance   = model.feature_importances_
top_idx      = np.argsort(importance)[::-1][:15]
axes[2].barh([feature_cols[i] for i in top_idx],
             importance[top_idx], color='#00d4aa')
axes[2].set_title('Top 15 Feature Importances')
axes[2].set_xlabel('Importance')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("\n✅ Saved training_results.png")
