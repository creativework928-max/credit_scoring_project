from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT /
    "data" /
    "processed" /
    "german_credit_processed.csv"
)

MODEL = (
    ROOT /
    "models" /
    "best_model.joblib"
)


df = pd.read_csv(DATA)

model = joblib.load(MODEL)

X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]


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


# UCI-inspired cost:
FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 5


threshold_results = []

for threshold in np.arange(
    0.05,
    0.96,
    0.01
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    cost = (
        FALSE_POSITIVE_COST * fp +
        FALSE_NEGATIVE_COST * fn
    )

    threshold_results.append({
        "threshold": threshold,
        "false_positives": fp,
        "false_negatives": fn,
        "cost": cost
    })


threshold_df = pd.DataFrame(
    threshold_results
)

best_row = threshold_df.loc[
    threshold_df["cost"].idxmin()
]

print("\nOptimal threshold:")
print(best_row)


threshold_df.to_csv(
    ROOT /
    "reports" /
    "threshold_analysis.csv",
    index=False
)
