import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "/Users/gabrieldiaz/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df[['Time','Amount','Class']].head())

print(f"\n=== CLASS DISTRIBUTION ===")
fraud    = df[df['Class']==1]
legit    = df[df['Class']==0]
print(f"Legitimate:  {len(legit):>6} ({len(legit)/len(df)*100:.2f}%)")
print(f"Fraud:       {len(fraud):>6} ({len(fraud)/len(df)*100:.2f}%)")
print(f"Fraud ratio: 1 in every {len(legit)//len(fraud)} transactions")

print(f"\n=== AMOUNT STATISTICS ===")
print(f"Fraud    — mean: ${fraud['Amount'].mean():.2f}  max: ${fraud['Amount'].max():.2f}")
print(f"Legit    — mean: ${legit['Amount'].mean():.2f}  max: ${legit['Amount'].max():.2f}")

print(f"\n=== FEATURE INFO ===")
print("V1-V28: PCA-transformed features (anonymized for privacy)")
print("Time:   Seconds elapsed since first transaction")
print("Amount: Transaction amount in euros")
print("Class:  0=legitimate, 1=fraud")

# Plot class imbalance + amount distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Class imbalance
axes[0].bar(['Legitimate','Fraud'], [len(legit), len(fraud)],
            color=['#00d4aa','#ff4444'])
axes[0].set_title('Class Distribution\n(Severe Imbalance)')
axes[0].set_ylabel('Count')
for i, v in enumerate([len(legit), len(fraud)]):
    axes[0].text(i, v + 1000, str(v), ha='center', fontweight='bold')

# Amount distribution
axes[1].hist(legit['Amount'].clip(upper=500),  bins=50,
             alpha=0.6, color='#00d4aa', label='Legitimate')
axes[1].hist(fraud['Amount'].clip(upper=500),  bins=50,
             alpha=0.6, color='#ff4444', label='Fraud')
axes[1].set_title('Transaction Amount Distribution\n(clipped at $500)')
axes[1].set_xlabel('Amount ($)')
axes[1].legend()

# Fraud over time
axes[2].scatter(fraud['Time']/3600, fraud['Amount'],
                alpha=0.3, color='#ff4444', s=5)
axes[2].set_title('Fraud Transactions Over Time')
axes[2].set_xlabel('Hours elapsed')
axes[2].set_ylabel('Amount ($)')

plt.tight_layout()
plt.savefig("data_exploration.png", dpi=150)
print("\n✅ Saved data_exploration.png")

