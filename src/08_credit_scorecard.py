from pathlib import Path

import joblib
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    ROOT /
    "models" /
    "best_model.joblib"
)


model = joblib.load(
    MODEL_FILE
)


def calculate_credit_score(
    applicant: pd.DataFrame
):

    probability_bad = model.predict_proba(
        applicant
    )[:, 1][0]

    # Map bad-risk probability into score.
    #
    # Lower bad probability -> higher score.
    #
    # 300 = very high risk
    # 850 = very low risk

    score = (
        850 -
        (probability_bad * 550)
    )

    score = float(
        np.clip(
            score,
            300,
            850
        )
    )

    if score >= 750:
        category = "Excellent"

    elif score >= 700:
        category = "Good"

    elif score >= 650:
        category = "Fair"

    elif score >= 600:
        category = "Below Average"

    else:
        category = "High Risk"

    return {
        "probability_bad": probability_bad,
        "credit_score": round(score),
        "risk_category": category
    }
