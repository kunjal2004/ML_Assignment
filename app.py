
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# ============================================================
# Bank Marketing | ML Experiment Lab
# Evaluation-only app: models are loaded, never retrained.
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "deposit"

LABEL_TO_INT = {"no": 0, "yes": 1}
CLASS_NAMES = ["No subscription", "Subscribed"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# Configure the Streamlit page before creating UI elements.
st.set_page_config(
    page_title="Bank Marketing | ML Experiment Lab",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
      }

      .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102a43, #245b75);
        color: white;
        margin-bottom: 1.2rem;
      }

      .hero h1 {
        font-size: 2rem;
        margin: 0 0 .4rem 0;
        color: white;
      }

      .hero p {
        margin: 0;
        color: #e6f1f5;
        font-size: 1rem;
      }

      div[data-testid="stMetric"] {
        border: 1px solid #d9e2ec;
        padding: 12px 14px;
        border-radius: 12px;
        background: #f8fafc;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div style="color:#b9dce8;font-size:.75rem;font-weight:700;letter-spacing:.08em;">
        CLASSIFICATION • EXPERIMENT WORKSPACE
      </div>
      <h1>Bank Marketing | ML Experiment Lab</h1>
      <p>
        Compare five saved classifiers on labelled test data
        and inspect the errors behind each score.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Evaluation-only workflow: the saved preprocessing pipeline and "
    "trained models are loaded from model/. Uploaded data is transformed "
    "and evaluated; models are not retrained."
)


# ============================================================
# Load the saved preprocessor and models
# ============================================================

@st.cache_resource
def load_saved_artifacts():
    """Load fitted preprocessing and trained models from disk."""

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_DIR}. "
            "Place the model/ folder beside app.py."
        )

    required_files = [
        "preprocessor.joblib",
        *MODEL_FILES.values(),
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (MODEL_DIR / filename).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing files in model/: " + ", ".join(missing_files)
        )

    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")

    models = {
        model_name: joblib.load(MODEL_DIR / filename)
        for model_name, filename in MODEL_FILES.items()
    }

    return preprocessor, models


def calculate_metrics(estimator, features, labels, dense_for_nb=False):
    """Calculate metrics and predictions for a saved classifier."""

    model_features = features

    # GaussianNB requires a dense array if preprocessing returns sparse data.
    if dense_for_nb and hasattr(features, "toarray"):
        model_features = features.toarray()

    predictions = estimator.predict(model_features)

    # Use probabilities where available; otherwise use decision scores.
    if hasattr(estimator, "predict_proba"):
        positive_scores = estimator.predict_proba(model_features)[:, 1]
    elif hasattr(estimator, "decision_function"):
        positive_scores = estimator.decision_function(model_features)
    else:
        positive_scores = None

    auc = np.nan

    if positive_scores is not None and len(np.unique(labels)) == 2:
        try:
            auc = roc_auc_score(labels, positive_scores)
        except (ValueError, TypeError):
            auc = np.nan

    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": auc,
        "Precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "Recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            labels, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(labels, predictions),
    }

    return metrics, predictions, positive_scores


# ============================================================
# Load model artifacts safely
# ============================================================

try:
    preprocessor, models = load_saved_artifacts()

except Exception as exc:
    st.error("The saved models could not be loaded.")
    st.code(str(exc))

    st.info(
        "Check that app.py is beside the model/ folder. "
        "The folder must contain all five saved model files "
        "and preprocessor.joblib."
    )

    st.stop()


# Read the input feature names from the fitted preprocessor.
expected_columns = list(
    getattr(preprocessor, "feature_names_in_", [])
)

if not expected_columns:
    st.error(
        "The saved preprocessor does not expose feature_names_in_. "
        "Ensure preprocessor.joblib is the fitted preprocessing "
        "pipeline used for the Bank Marketing dataset."
    )
    st.stop()


# ============================================================
# 1. Upload and validate labelled test data
# ============================================================

st.subheader("1 · Load a labelled test dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help=(
        "The CSV must contain all original input features "
        "and the deposit target column."
    ),
)

if uploaded_file is None:
    st.info("Upload test_data.csv to start the model comparison.")
    st.stop()

try:
    uploaded_data = pd.read_csv(uploaded_file)

except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if uploaded_data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

if TARGET_COLUMN not in uploaded_data.columns:
    st.error(
        f"The uploaded CSV must contain the '{TARGET_COLUMN}' column."
    )
    st.stop()

missing_columns = [
    column
    for column in expected_columns
    if column not in uploaded_data.columns
]

if missing_columns:
    st.error(
        "The CSV is missing required input features: "
        + ", ".join(missing_columns)
    )
    st.stop()

# Normalize target labels and map them to integers.
raw_labels = (
    uploaded_data[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)

unexpected_labels = sorted(
    set(raw_labels.unique()) - set(LABEL_TO_INT)
)

if unexpected_labels:
    st.error(
        "The target column must contain only 'yes' and 'no'. "
        f"Unexpected values: {unexpected_labels}"
    )
    st.stop()

if raw_labels.nunique() < 2:
    st.warning(
        "The uploaded file contains only one target class. "
        "AUC cannot be meaningfully calculated for this file."
    )

X_raw = uploaded_data[expected_columns].copy()
y_true = raw_labels.map(LABEL_TO_INT).astype(int)

# Display basic dataset information.
info_col1, info_col2, info_col3 = st.columns(3)

info_col1.metric("Test rows", f"{len(uploaded_data):,}")
info_col2.metric("Input features", str(len(expected_columns)))
info_col3.metric("Target classes", str(raw_labels.nunique()))

with st.expander("Preview uploaded data"):
    st.dataframe(
        uploaded_data.head(10),
        use_container_width=True,
    )

# Transform only. Do not fit or retrain the preprocessor.
try:
    X_transformed = preprocessor.transform(X_raw)

except Exception as exc:
    st.error(f"Preprocessing failed for the uploaded data: {exc}")
    st.stop()

st.success(
    "Data validated and transformed using the saved preprocessing pipeline."
)


# ============================================================
# 2. Compare all saved models
# ============================================================

st.subheader("2 · Model comparison")

st.write(
    "All metrics are calculated on the uploaded labelled test data. "
    "The saved models are loaded from disk and are not retrained."
)

all_results = {}
all_predictions = {}
evaluation_errors = {}

for model_name, estimator in models.items():
    try:
        metrics, predictions, _ = calculate_metrics(
            estimator=estimator,
            features=X_transformed,
            labels=y_true,
            dense_for_nb=(model_name == "Gaussian Naive Bayes"),
        )

        all_results[model_name] = metrics
        all_predictions[model_name] = predictions

    except Exception as exc:
        evaluation_errors[model_name] = str(exc)

if evaluation_errors:
    st.warning("One or more models could not be evaluated.")

    for model_name, message in evaluation_errors.items():
        st.write(f"**{model_name}:** {message}")

if not all_results:
    st.error("No model could be evaluated. Check the saved model files.")
    st.stop()

# Rank models by MCC for display.
comparison = pd.DataFrame.from_dict(
    all_results,
    orient="index",
)

comparison.index.name = "Model"
comparison = comparison.reset_index()

comparison = comparison.sort_values(
    by="MCC",
    ascending=False,
    na_position="last",
)

comparison_display = comparison.copy()

for metric_name in [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1-score",
    "MCC",
]:
    comparison_display[metric_name] = comparison_display[
        metric_name
    ].map(
        lambda value: (
            "—" if pd.isna(value) else f"{value:.4f}"
        )
    )

st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True,
)

best_model_name = comparison.iloc[0]["Model"]
best_row = comparison.iloc[0]

best_col1, best_col2, best_col3 = st.columns(3)

best_col1.metric("Top MCC model", best_model_name)
best_col2.metric(
    "Best model accuracy",
    f"{best_row['Accuracy']:.4f}",
)
best_col3.metric(
    "Best model AUC",
    "—"
    if pd.isna(best_row["AUC"])
    else f"{best_row['AUC']:.4f}",
)

st.caption(
    "The model at the top is ranked by MCC on this uploaded dataset. "
    "The top-ranked model may change with different evaluation data."
)


# ============================================================
# 3. Inspect one model in detail
# ============================================================

st.subheader("3 · Inspect a model")

available_models = list(all_results.keys())

selected_model_name = st.selectbox(
    "Choose a model to inspect",
    options=available_models,
    index=available_models.index(best_model_name),
)

selected_metrics = all_results[selected_model_name]
selected_predictions = all_predictions[selected_model_name]

metric_order = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1-score",
    "MCC",
]

metric_columns = st.columns(3)

for index, metric_name in enumerate(metric_order):
    value = selected_metrics[metric_name]

    formatted_value = (
        "—" if pd.isna(value) else f"{value:.4f}"
    )

    metric_columns[index % 3].metric(
        metric_name,
        formatted_value,
    )


# ============================================================
# Confusion matrix and classification report
# ============================================================

matrix_col, report_col = st.columns([1, 1])

with matrix_col:
    st.markdown("#### Confusion matrix")

    matrix = confusion_matrix(
        y_true,
        selected_predictions,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    ).plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )

    axis.set_title(selected_model_name)
    figure.tight_layout()

    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

with report_col:
    st.markdown("#### Classification report")

    report = classification_report(
        y_true,
        selected_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_table = pd.DataFrame(report).transpose()

    st.dataframe(
        report_table.round(4),
        use_container_width=True,
    )


# ============================================================
# 4. Review and download predictions
# ============================================================

st.subheader("4 · Prediction review")

st.write(
    "Inspect where the selected model agrees or disagrees "
    "with the actual labels in the uploaded CSV."
)

prediction_review = uploaded_data.copy()

prediction_review["Actual deposit"] = (
    y_true.map({0: "no", 1: "yes"}).to_numpy()
)

prediction_review["Predicted deposit"] = pd.Series(
    selected_predictions,
    index=uploaded_data.index,
).map({0: "no", 1: "yes"})

prediction_review["Correct prediction"] = (
    prediction_review["Actual deposit"]
    == prediction_review["Predicted deposit"]
)

show_only_errors = st.checkbox(
    "Show only incorrect predictions",
    value=False,
)

if show_only_errors:
    prediction_review = prediction_review.loc[
        ~prediction_review["Correct prediction"]
    ]

st.dataframe(
    prediction_review.head(100),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "A maximum of 100 rows is displayed to keep the interface responsive."
)

csv_bytes = prediction_review.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download displayed prediction review",
    data=csv_bytes,
    file_name="bank_marketing_prediction_review.csv",
    mime="text/csv",
)


# ============================================================
# 5. Metric explanations
# ============================================================

with st.expander("How to interpret the metrics"):
    st.markdown(
        """
        - **Accuracy:** Proportion of all predictions that are correct.
        - **AUC:** How well the model ranks positive cases above negative cases.
        - **Precision:** Among customers predicted to subscribe, the proportion who did.
        - **Recall:** Among customers who actually subscribed, the proportion identified.
        - **F1-score:** Harmonic mean of precision and recall.
        - **MCC:** Summary of prediction quality using all confusion-matrix
          categories. It ranges from -1 to +1; higher is generally better.

        AUC is unavailable when the uploaded test file contains only one
        target class.
        """
    )
