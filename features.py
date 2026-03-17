import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = "/Users/gabrieldiaz/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# ── Feature Engineering ───────────────────────────────────────────────────────
print("Engineering features...")

# Time-based features
df['hour_of_day']  = (df['Time'] / 3600) % 24
df['is_night']     = ((df['hour_of_day'] >= 23) | (df['hour_of_day'] <= 5)).astype(int)

# Amount-based features
df['amount_log']   = np.log1p(df['Amount'])
df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df['is_round_amount'] = (df['Amount'] % 1 == 0).astype(int)
df['is_large']     = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)

# Velocity features — transactions in same time window
# Sort by time, compute rolling counts (proxy for card velocity)
df = df.sort_values('Time').reset_index(drop=True)
df['tx_per_hour_window'] = df.groupby(
    pd.cut(df['Time'], bins=100))['Time'].transform('count')
df['tx_per_hour_window'] = df['tx_per_hour_window'].fillna(1)

# V feature interactions — top fraud-correlated PCA components
# V14, V10, V12, V4, V11 are most correlated with fraud historically
df['v14_v10']  = df['V14'] * df['V10']
df['v12_v4']   = df['V12'] * df['V4']
df['v14_sq']   = df['V14'] ** 2
df['v10_sq']   = df['V10'] ** 2
df['v_fraud_score'] = (
    -0.3 * df['V14'] - 0.25 * df['V10'] -
     0.2 * df['V12'] + 0.2 * df['V4'] +
     0.15 * df['V11']
)

FEATURE_COLS = (
    [f'V{i}' for i in range(1, 29)] +
    ['amount_log', 'amount_zscore', 'is_round_amount',
     'is_large', 'hour_of_day', 'is_night',
     'tx_per_hour_window', 'v14_v10', 'v12_v4',
     'v14_sq', 'v10_sq', 'v_fraud_score']
)

print(f"Total features: {len(FEATURE_COLS)}")

X = df[FEATURE_COLS].values.astype(np.float32)
y = df['Class'].values.astype(int)

# ── Train/test split — stratified to preserve fraud ratio ────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {X_train.shape} | Fraud: {y_train.sum()}")
print(f"Test:  {X_test.shape}  | Fraud: {y_test.sum()}")

# Save
np.save("X_train.npy", X_train)
np.save("X_test.npy",  X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy",  y_test)
joblib.dump(scaler,       "scaler_fraud.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")

print("✅ Saved all feature files")
