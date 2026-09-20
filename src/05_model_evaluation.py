from pathlib import Path

import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

from sklearn.calibration import calibration_curve


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT /
    "data" /
    "processed" /
    "german_credit_processed.csv"
)

MODEL_FILE = (
    ROOT /
    "models" /
    "best_model.joblib"
)

FIG_DIR = (
    ROOT /
    "reports" /
    "figures"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_FILE)

model = joblib.load(MODEL_FILE)

TARGET = "credit_risk"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# IMPORTANT:
# Use the same fixed test split as training.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    predictions,
    target_names=[
        "Good Credit",
        "Bad Credit"
    ]
)

print(report)

REPORT_DIR = ROOT / "reports"

with open(
    REPORT_DIR /
    "classification_report.txt",
    "w"
) as f:
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=[
        "Good",
        "Bad"
    ],
    yticklabels=[
        "Good",
        "Bad"
    ]
)

plt.title(
    "Credit Risk Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "09_confusion_matrix.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities
)

auc = roc_auc_score(
    y_test,
    probabilities
)

plt.figure(figsize=(9, 7))

plt.plot(
    fpr,
    tpr,
    color="#2563EB",
    linewidth=3,
    label=f"Model ROC-AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="#94A3B8"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve",
    fontweight="bold"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "10_roc_curve.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision, recall, pr_thresholds = (
    precision_recall_curve(
        y_test,
        probabilities
    )
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

plt.figure(figsize=(9, 7))

plt.plot(
    recall,
    precision,
    color="#0F766E",
    linewidth=3,
    label=f"PR-AUC = {pr_auc:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve",
    fontweight="bold"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "11_precision_recall_curve.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CALIBRATION
# ============================================================

fraction_positive, mean_predicted = (
    calibration_curve(
        y_test,
        probabilities,
        n_bins=10
    )
)

brier = brier_score_loss(
    y_test,
    probabilities
)

plt.figure(figsize=(9, 7))

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    linewidth=2.5,
    color="#2563EB",
    label="Credit Model"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="#94A3B8",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Bad-Credit Rate")

plt.title(
    f"Probability Calibration | Brier Score = {brier:.3f}",
    fontweight="bold"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "12_calibration_curve.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COST MATRIX
# ============================================================

# UCI cost matrix:
#
# Actual Good / Predicted Good = 0
# Actual Good / Predicted Bad  = 1
# Actual Bad  / Predicted Good = 5
# Actual Bad  / Predicted Bad  = 0

tn, fp, fn, tp = cm.ravel()

credit_cost = (
    1 * fp +
    5 * fn
)

print("\nCredit-risk cost:")
print(credit_cost)

print("\nROC-AUC:", auc)
print("PR-AUC:", pr_auc)
print("Brier Score:", brier)
print("Cost:", credit_cost)
