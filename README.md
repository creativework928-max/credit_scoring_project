# Credit Intelligence Platform

## End-to-End Credit Scoring Machine Learning Project

This project develops a machine-learning system for predicting
creditworthiness.

The primary dataset is the UCI Statlog German Credit dataset.

---

## Objective

Predict whether an applicant represents:

- Good Credit Risk
- Bad Credit Risk

The project additionally produces a project-specific credit score
between 300 and 850.

---

## Dataset

Source:

UCI Machine Learning Repository

Statlog German Credit Data

Dataset characteristics:

- 1,000 observations
- 20 predictors
- Binary target
- Categorical and numerical variables
- No missing values in the original UCI representation

Important:

The dataset does not contain annual income.

No artificial income values are created.

Financial capacity is represented through available financial
variables such as credit amount, installment burden,
checking-account status, savings and employment.

---

## Project Workflow

1. Data acquisition
2. Data validation
3. Exploratory data analysis
4. Feature engineering
5. Data preprocessing
6. Train/test splitting
7. Model training
8. Model comparison
9. Model evaluation
10. Threshold optimization
11. Explainability
12. Credit score construction
13. Applicant prediction
14. Streamlit deployment
15. Automated report generation

---

## Models

The project compares:

- Logistic Regression
- Random Forest
- Extra Trees
- HistGradientBoosting

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Brier Score
- Cost-sensitive evaluation

---

## Installation

```bash
pip install -r requirements.txt
