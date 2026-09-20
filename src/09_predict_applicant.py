from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    ROOT /
    "models" /
    "best_model.joblib"
)

model = joblib.load(
    MODEL_FILE
)


def predict_applicant(applicant_dict):

    applicant = pd.DataFrame([
        applicant_dict
    ])

    probability_bad = (
        model
        .predict_proba(applicant)[:, 1][0]
    )

    probability_good = (
        1 - probability_bad
    )

    prediction = (
        "Bad Credit"
        if probability_bad >= 0.50
        else "Good Credit"
    )

    return {
        "prediction": prediction,
        "probability_good": round(
            probability_good,
            4
        ),
        "probability_bad": round(
            probability_bad,
            4
        )
    }
