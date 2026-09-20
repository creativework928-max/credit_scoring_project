from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw" / "german_credit_raw.csv"
OUTPUT = ROOT / "data" / "processed" / "german_credit_processed.csv"


df = pd.read_csv(INPUT)


# ============================================================
# FEATURE NAMES
# ============================================================

df = df.rename(columns={
    "Attribute1": "checking_account",
    "Attribute2": "loan_duration_months",
    "Attribute3": "credit_history",
    "Attribute4": "loan_purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_account",
    "Attribute7": "employment",
    "Attribute8": "installment_rate",
    "Attribute9": "personal_status_sex",
    "Attribute10": "other_debtors",
    "Attribute11": "residence_years",
    "Attribute12": "property",
    "Attribute13": "age",
    "Attribute14": "other_installment_plans",
    "Attribute15": "housing",
    "Attribute16": "existing_credits",
    "Attribute17": "job",
    "Attribute18": "dependents",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker"
})


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Loan amount relative to duration.
df["credit_amount_per_month"] = (
    df["credit_amount"] /
    df["loan_duration_months"].replace(0, np.nan)
)

# Age buckets.
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 50, 65, 100],
    labels=[
        "18-25",
        "26-35",
        "36-50",
        "51-65",
        "65+"
    ],
    include_lowest=True
)

# Loan duration buckets.
df["duration_group"] = pd.cut(
    df["loan_duration_months"],
    bins=[0, 12, 24, 36, 60],
    labels=[
        "<=12 months",
        "13-24 months",
        "25-36 months",
        "37-60 months"
    ],
    include_lowest=True
)

# High installment burden indicator.
df["high_installment_burden"] = (
    df["installment_rate"] >= 4
).astype(int)

# Multiple existing-credit indicator.
df["multiple_existing_credits"] = (
    df["existing_credits"] >= 2
).astype(int)

# Long employment indicator based on UCI categories.
df["stable_employment"] = (
    df["employment"].isin(["A74", "A75"])
).astype(int)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicates:", df.duplicated().sum())


# Remove exact duplicate records if any.
df = df.drop_duplicates().reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT,
    index=False
)

print("\nProcessed dataset saved to:")
print(OUTPUT)

print("\nFinal shape:", df.shape)
