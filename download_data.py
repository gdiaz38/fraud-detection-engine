import kagglehub
import os

print("Downloading Credit Card Fraud Dataset...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print(f"Downloaded to: {path}")

for root, dirs, files in os.walk(path):
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath) / 1024
        print(f"  {filepath} ({size:.1f} KB)")
