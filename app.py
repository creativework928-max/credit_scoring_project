import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Intelligence",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F8FAFC;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 18px rgba(15, 23, 42, 0.08);
        border: 1px solid #E2E8F0;
    }

    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 15px;
        border-radius: 12px;
        font-weight: bold;
    }

    .risk-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 15px;
        border-radius: 12px;
        font-weight: bold;
    }

    h1 {
        color: #102A43;
    }

    h2 {
        color: #102A43;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = (
    "models/best_model.joblib"
)

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "💳 Credit Intelligence Platform"
)

st.caption(
    "Machine-learning based credit-risk assessment"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Applicant Information"
)


# UCI categories.

checking_account = st.sidebar.selectbox(
    "Checking Account",
    [
        "A11",
        "A12",
        "A13",
        "A14"
    ]
)

loan_duration = st.sidebar.slider(
    "Loan Duration (months)",
    4,
    72,
    24
)

credit_history = st.sidebar.selectbox(
    "Credit History",
    [
        "A30",
        "A31",
        "A32",
        "A33",
        "A34"
    ]
)

loan_purpose = st.sidebar.selectbox(
    "Loan Purpose",
    [
        "A40",
        "A41",
        "A42",
        "A43",
        "A44",
        "A45",
        "A46",
        "A47",
        "A48",
        "A49",
        "A410"
    ]
)

credit_amount = st.sidebar.number_input(
    "Credit Amount",
    min_value=250,
    max_value=25000,
    value=4000
)

savings = st.sidebar.selectbox(
    "Savings Account",
    [
        "A61",
        "A62",
        "A63",
        "A64",
        "A65"
    ]
)

employment = st.sidebar.selectbox(
    "Employment",
    [
        "A71",
        "A72",
        "A73",
        "A74",
        "A75"
    ]
)

installment_rate = st.sidebar.slider(
    "Installment Rate",
    1,
    4,
    2
)

personal_status = st.sidebar.selectbox(
    "Personal Status / Sex",
    [
        "A91",
        "A92",
        "A93",
        "A94",
        "A95"
    ]
)

other_debtors = st.sidebar.selectbox(
    "Other Debtors",
    [
        "A101",
        "A102",
        "A103"
    ]
)

residence_years = st.sidebar.slider(
    "Residence Years",
    1,
    4,
    2
)

property_type = st.sidebar.selectbox(
    "Property",
    [
        "A121",
        "A122",
        "A123",
        "A124"
    ]
)

age = st.sidebar.slider(
    "Age",
    18,
    75,
    35
)

other_plans = st.sidebar.selectbox(
    "Other Installment Plans",
    [
        "A141",
        "A142",
        "A143"
    ]
)

housing = st.sidebar.selectbox(
    "Housing",
    [
        "A151",
        "A152",
        "A153"
    ]
)

existing_credits = st.sidebar.slider(
    "Existing Credits",
    1,
    4,
    1
)

job = st.sidebar.selectbox(
    "Job",
    [
        "A171",
        "A172",
        "A173",
        "A174"
    ]
)

dependents = st.sidebar.slider(
    "Dependents",
    1,
    2,
    1
)

telephone = st.sidebar.selectbox(
    "Telephone",
    [
        "A191",
        "A192"
    ]
)

foreign_worker = st.sidebar.selectbox(
    "Foreign Worker",
    [
        "A201",
        "A202"
    ]
)


# ============================================================
# BUILD APPLICANT
# ============================================================

applicant = pd.DataFrame([{

    "checking_account":
        checking_account,

    "loan_duration_months":
        loan_duration,

    "credit_history":
        credit_history,

    "loan_purpose":
        loan_purpose,

    "credit_amount":
        credit_amount,

    "savings_account":
        savings,

    "employment":
        employment,

    "installment_rate":
        installment_rate,

    "personal_status_sex":
        personal_status,

    "other_debtors":
        other_debtors,

    "residence_years":
        residence_years,

    "property":
        property_type,

    "age":
        age,

    "other_installment_plans":
        other_plans,

    "housing":
        housing,

    "existing_credits":
        existing_credits,

    "job":
        job,

    "dependents":
        dependents,

    "telephone":
        telephone,

    "foreign_worker":
        foreign_worker,

    "credit_amount_per_month":
        credit_amount / loan_duration,

    "age_group":
        (
            "18-25" if age <= 25
            else "26-35" if age <= 35
            else "36-50" if age <= 50
            else "51-65" if age <= 65
            else "65+"
        ),

    "duration_group":
        (
            "<=12 months"
            if loan_duration <= 12
            else "13-24 months"
            if loan_duration <= 24
            else "25-36 months"
            if loan_duration <= 36
            else "37-60 months"
        ),

    "high_installment_burden":
        int(installment_rate >= 4),

    "multiple_existing_credits":
        int(existing_credits >= 2),

    "stable_employment":
        int(
            employment in [
                "A74",
                "A75"
            ]
        )
}])


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Assess Credit Risk",
    type="primary",
    use_container_width=True
):

    probability_bad = (
        model
        .predict_proba(applicant)[:, 1][0]
    )

    probability_good = (
        1 - probability_bad
    )

    score = (
        850 -
        probability_bad * 550
    )

    score = np.clip(
        score,
        300,
        850
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


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Credit Score",
            f"{score:.0f}"
        )

    with col2:

        st.metric(
            "Good-Credit Probability",
            f"{probability_good:.1%}"
        )

    with col3:

        st.metric(
            "Bad-Credit Probability",
            f"{probability_bad:.1%}"
        )


    st.divider()


    # ========================================================
    # DECISION
    # ========================================================

    if probability_bad < 0.50:

        st.markdown(
            """
            <div class="risk-low">
            ✓ LOWER RISK — Applicant classified as Good Credit
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="risk-high">
            ⚠ HIGHER RISK — Applicant classified as Bad Credit
            </div>
            """,
            unsafe_allow_html=True
        )


    st.subheader(
        "Risk Profile"
    )

    risk_df = pd.DataFrame({
        "Risk": [
            "Good Credit",
            "Bad Credit"
        ],
        "Probability": [
            probability_good,
            probability_bad
        ]
    })

    st.bar_chart(
        risk_df.set_index("Risk")
    )


    st.info(
        f"Risk category: **{category}**"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Educational / analytical model only. "
    "This score is not an official banking or regulatory credit score "
    "and should not be used as the sole basis for real lending decisions."
)
