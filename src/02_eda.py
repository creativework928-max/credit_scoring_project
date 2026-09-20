from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "raw" / "german_credit_raw.csv"
FIG_DIR = ROOT / "reports" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROFESSIONAL STYLE
# ============================================================

COLORS = {
    "navy": "#102A43",
    "blue": "#2563EB",
    "teal": "#0F766E",
    "green": "#16A34A",
    "orange": "#F59E0B",
    "red": "#DC2626",
    "light": "#F3F6FA",
    "gray": "#64748B",
    "dark": "#172033"
}

sns.set_theme(
    style="whitegrid",
    font_scale=1.05
)

plt.rcParams.update({
    "figure.figsize": (11, 7),
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "figure.dpi": 140
})


df = pd.read_csv(DATA_FILE)


# ============================================================
# 1. TARGET DISTRIBUTION
# ============================================================

target_counts = df["credit_risk"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 6))

labels = ["Good Credit", "Bad Credit"]

bars = ax.bar(
    labels,
    target_counts.values,
    color=[COLORS["green"], COLORS["red"]],
    width=0.55
)

ax.set_title("Credit Risk Distribution")
ax.set_ylabel("Number of Applicants")
ax.set_xlabel("Credit Risk")

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(bar.get_height()):,}",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

sns.despine()

plt.tight_layout()
plt.savefig(
    FIG_DIR / "01_target_distribution.png",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. CREDIT AMOUNT BY RISK
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

sns.boxplot(
    data=df,
    x="credit_risk",
    y="Attribute5",
    hue="credit_risk",
    palette=[COLORS["green"], COLORS["red"]],
    legend=False,
    ax=ax
)

ax.set_xticklabels(["Good Credit", "Bad Credit"])
ax.set_xlabel("Credit Risk")
ax.set_ylabel("Credit Amount")
ax.set_title("Credit Amount Distribution by Credit Risk")

sns.despine()

plt.tight_layout()
plt.savefig(
    FIG_DIR / "02_credit_amount_risk.png",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. AGE DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

sns.histplot(
    data=df,
    x="Attribute13",
    hue="credit_risk",
    bins=20,
    kde=True,
    palette=[COLORS["green"], COLORS["red"]],
    alpha=0.45,
    ax=ax
)

ax.set_title("Age Distribution by Credit Risk")
ax.set_xlabel("Age")
ax.set_ylabel("Number of Applicants")

sns.despine()

plt.tight_layout()
plt.savefig(
    FIG_DIR / "03_age_distribution.png",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. LOAN DURATION
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

sns.boxplot(
    data=df,
    x="credit_risk",
    y="Attribute2",
    hue="credit_risk",
    palette=[COLORS["green"], COLORS["red"]],
    legend=False,
    ax=ax
)

ax.set_xticklabels(["Good Credit", "Bad Credit"])
ax.set_xlabel("Credit Risk")
ax.set_ylabel("Loan Duration (months)")
ax.set_title("Loan Duration by Credit Risk")

sns.despine()

plt.tight_layout()
plt.savefig(
    FIG_DIR / "04_loan_duration.png",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. INSTALLMENT RATE
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

sns.countplot(
    data=df,
    x="Attribute8",
    hue="credit_risk",
    palette=[COLORS["green"], COLORS["red"]],
    ax=ax
)

ax.set_title("Installment Rate by Credit Risk")
ax.set_xlabel("Installment Rate (% of Disposable Income)")
ax.set_ylabel("Applicants")

sns.despine()

plt.tight_layout()
plt.savefig(
    FIG_DIR / "05_installment_rate.png",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. CREDIT HISTORY
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

risk_rate = (
    df.groupby("Attribute3")["credit_risk"]
    .mean()
    .sort_values(ascending=False)
)

bars = ax.bar(
    risk_rate.index.astype(str),
    risk_rate.values * 100,
    color=COLORS["blue"]
)

ax.set_title("Bad-Credit Rate by Credit History")
ax.set_xlabel("Credit History Category")
ax.set_ylabel("Bad-Credit Rate (%)")

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.1f}%",
        ha="center",
        va="bottom"
    )

sns.despine()

plt.tight_layout()
plt.savefig(
    FIG_DIR / "06_credit_history_risk.png",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

numeric_cols = df.select_dtypes(include=np.number).columns

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(
    corr,
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

ax.set_title("Numeric Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    FIG_DIR / "07_correlation_matrix.png",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. SAVINGS VS RISK
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

risk_by_savings = (
    df.groupby("Attribute6")["credit_risk"]
    .mean()
    .sort_values(ascending=False)
)

bars = ax.bar(
    risk_by_savings.index.astype(str),
    risk_by_savings.values * 100,
    color=COLORS["orange"]
)

ax.set_title("Bad-Credit Rate by Savings Category")
ax.set_xlabel("Savings Category")
ax.set_ylabel("Bad-Credit Rate (%)")

plt.xticks(rotation=25)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.1f}%",
        ha="center",
        va="bottom"
    )

sns.despine()

plt.tight_layout()

plt.savefig(
    FIG_DIR / "08_savings_risk.png",
    bbox_inches="tight"
)

plt.close()


print("EDA completed.")
print(f"Figures saved to: {FIG_DIR}")
