from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOWNLOAD DATA
# ============================================================

print("=" * 70)
print("CREDIT SCORING PROJECT - DATA COLLECTION")
print("=" * 70)

dataset = fetch_ucirepo(id=144)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget:")
print(y.head())


# ============================================================
# COMBINE FEATURES AND TARGET
# ============================================================

target_column = y.columns[0]

df = X.copy()
df["credit_risk"] = y[target_column]


# ============================================================
# STANDARDIZE TARGET
#
# UCI:
# 1 = Good
# 2 = Bad
#
# We convert:
# Good -> 0
# Bad  -> 1
# ============================================================

df["credit_risk"] = df["credit_risk"].map({
    1: 0,
    2: 1,
    "1": 0,
    "2": 1
})


# ============================================================
# SAVE RAW DATA
# ============================================================

output_file = RAW_DIR / "german_credit_raw.csv"

df.to_csv(output_file, index=False)

print("\nSaved:", output_file)

print("\nTarget distribution:")
print(df["credit_risk"].value_counts())

print("\nTarget percentages:")
print(df["credit_risk"].value_counts(normalize=True).mul(100).round(2))

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isna().sum())

print("\nData collection completed.")
