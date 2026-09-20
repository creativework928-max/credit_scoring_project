from pathlib import Path

import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import shap


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "german_credit_processed.csv"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "best_model.joblib"
)

FIG_DIR = (
    ROOT
    / "reports"
    / "figures"
)

REPORT_DIR = (
    ROOT
    / "reports"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREDIT SCORING - SHAP EXPLAINABILITY")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

TARGET = "credit_risk"

X = df.drop(
    columns=[TARGET]
)

print("\nDataset shape:", X.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")


# ============================================================
# EXTRACT PREPROCESSOR AND ESTIMATOR
# ============================================================

preprocessor = model.named_steps[
    "preprocessing"
]

estimator = model.named_steps[
    "model"
]


# ============================================================
# TRANSFORM FEATURES
# ============================================================

print("\nTransforming features...")

X_transformed = preprocessor.transform(
    X
)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

feature_names = np.asarray(
    feature_names
).ravel()

print(
    "Transformed feature shape:",
    X_transformed.shape
)

print(
    "Number of feature names:",
    len(feature_names)
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

if hasattr(
    X_transformed,
    "toarray"
):
    X_transformed = (
        X_transformed.toarray()
    )

X_transformed = np.asarray(
    X_transformed
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

try:

    # Tree models
    if hasattr(
        estimator,
        "feature_importances_"
    ):

        print(
            "Detected tree-based model."
        )

        explainer = shap.TreeExplainer(
            estimator
        )

        shap_output = explainer.shap_values(
            X_transformed
        )

    # Linear models
    else:

        print(
            "Detected non-tree model."
        )

        explainer = shap.Explainer(
            estimator,
            X_transformed
        )

        shap_output = explainer(
            X_transformed
        )

        # SHAP Explanation object
        if hasattr(
            shap_output,
            "values"
        ):

            shap_output = (
                shap_output.values
            )


except Exception as e:

    print(
        "\nSHAP TreeExplainer failed."
    )

    print(
        "Trying generic SHAP Explainer..."
    )

    explainer = shap.Explainer(
        estimator,
        X_transformed
    )

    shap_output = explainer(
        X_transformed
    )

    if hasattr(
        shap_output,
        "values"
    ):

        shap_output = (
            shap_output.values
        )


# ============================================================
# CONVERT SHAP OUTPUT
# ============================================================

print(
    "\nOriginal SHAP output type:",
    type(shap_output)
)

if isinstance(
    shap_output,
    list
):

    print(
        "SHAP returned a list."
    )

    print(
        "Number of outputs:",
        len(shap_output)
    )

    # For binary classification,
    # index 1 corresponds to the positive class
    if len(shap_output) == 2:

        shap_values = np.asarray(
            shap_output[1]
        )

    else:

        shap_values = np.asarray(
            shap_output[0]
        )

else:

    shap_values = np.asarray(
        shap_output
    )


print(
    "SHAP array shape:",
    shap_values.shape
)


# ============================================================
# HANDLE DIFFERENT SHAP DIMENSIONS
# ============================================================

if shap_values.ndim == 1:

    # Unexpected but possible for one observation
    shap_values = (
        shap_values.reshape(
            1,
            -1
        )
    )


elif shap_values.ndim == 2:

    # Standard format:
    #
    # samples x features
    #
    pass


elif shap_values.ndim == 3:

    print(
        "\n3-dimensional SHAP output detected."
    )

    print(
        "Shape:",
        shap_values.shape
    )

    # Common formats:
    #
    # samples x features x classes
    #
    # OR
    #
    # samples x classes x features
    #
    #
    # Determine which dimension corresponds
    # to the number of transformed features.

    n_features = len(
        feature_names
    )

    if shap_values.shape[1] == n_features:

        # samples x features x classes
        print(
            "Detected format: "
            "samples × features × classes"
        )

        if shap_values.shape[2] >= 2:

            # Positive / bad-credit class
            shap_values = (
                shap_values[:, :, 1]
            )

        else:

            shap_values = (
                shap_values[:, :, 0]
            )

    elif shap_values.shape[2] == n_features:

        # samples x classes x features
        print(
            "Detected format: "
            "samples × classes × features"
        )

        if shap_values.shape[1] >= 2:

            shap_values = (
                shap_values[:, 1, :]
            )

        else:

            shap_values = (
                shap_values[:, 0, :]
            )

    else:

        raise ValueError(
            "Unable to identify the feature "
            "dimension in SHAP output. "
            f"SHAP shape={shap_values.shape}, "
            f"number of features={n_features}"
        )

else:

    raise ValueError(
        "Unsupported SHAP output dimensions: "
        f"{shap_values.ndim}"
    )


# ============================================================
# FINAL SHAP SHAPE VALIDATION
# ============================================================

print(
    "\nFinal SHAP shape:",
    shap_values.shape
)

print(
    "Feature names shape:",
    feature_names.shape
)


if shap_values.ndim != 2:

    raise ValueError(
        "SHAP values must be 2-dimensional "
        "after processing."
    )


if shap_values.shape[1] != len(
    feature_names
):

    raise ValueError(
        "Mismatch between SHAP features "
        "and feature names.\n"
        f"SHAP features: {shap_values.shape[1]}\n"
        f"Feature names: {len(feature_names)}"
    )


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

print(
    "\nCalculating global feature importance..."
)

mean_abs_shap = np.abs(
    shap_values
).mean(
    axis=0
)


# Ensure 1-dimensional output
mean_abs_shap = np.asarray(
    mean_abs_shap
).ravel()


feature_names = np.asarray(
    feature_names
).ravel()


print(
    "Importance shape:",
    mean_abs_shap.shape
)


# ============================================================
# CREATE IMPORTANCE DATAFRAME
# ============================================================

importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})


importance = (
    importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = (
    REPORT_DIR /
    "shap_feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False
)

print(
    "\nFeature importance saved:"
)

print(
    importance_file
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\nTop 20 credit-risk drivers:")

print(
    importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# PROFESSIONAL SHAP BAR CHART
# ============================================================

top = (
    importance
    .head(20)
    .sort_values(
        "mean_abs_shap",
        ascending=True
    )
)


plt.figure(
    figsize=(12, 9)
)

bars = plt.barh(
    top["feature"],
    top["mean_abs_shap"],
    color="#2563EB"
)


# Add value labels
for bar in bars:

    width = bar.get_width()

    plt.text(
        width,
        bar.get_y()
        + bar.get_height() / 2,
        f" {width:.3f}",
        va="center",
        fontsize=9,
        color="#172033"
    )


plt.title(
    "Top Credit-Risk Drivers",
    fontsize=18,
    fontweight="bold",
    color="#102A43",
    pad=18
)

plt.xlabel(
    "Mean Absolute SHAP Value",
    fontsize=12
)

plt.ylabel(
    "Feature",
    fontsize=12
)


plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.25
)

plt.gca().spines[
    "top"
].set_visible(False)

plt.gca().spines[
    "right"
].set_visible(False)

plt.gca().spines[
    "left"
].set_visible(False)


plt.tight_layout()


shap_bar_file = (
    FIG_DIR /
    "13_shap_feature_importance.png"
)

plt.savefig(
    shap_bar_file,
    dpi=200,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


print(
    "\nSHAP bar chart saved:"
)

print(
    shap_bar_file
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print(
    "\nCreating SHAP summary plot..."
)

plt.figure(
    figsize=(12, 9)
)

try:

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_size=None
    )

    plt.title(
        "SHAP Summary — Credit Risk Model",
        fontsize=17,
        fontweight="bold",
        color="#102A43",
        pad=18
    )

    plt.tight_layout()

    summary_file = (
        FIG_DIR /
        "14_shap_summary.png"
    )

    plt.savefig(
        summary_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(
        "SHAP summary saved:"
    )

    print(
        summary_file
    )

except Exception as e:

    plt.close()

    print(
        "\nWarning: SHAP summary plot "
        "could not be generated."
    )

    print(
        "Reason:",
        e
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)

print(
    "SHAP EXPLAINABILITY COMPLETED SUCCESSFULLY"
)

print("=" * 70)

print(
    "\nGenerated files:"
)

print(
    "1.",
    importance_file
)

print(
    "2.",
    shap_bar_file
)

if (
    FIG_DIR /
    "14_shap_summary.png"
).exists():

    print(
        "3.",
        FIG_DIR /
        "14_shap_summary.png"
    )
