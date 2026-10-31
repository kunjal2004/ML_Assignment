from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ============================================================
# Bank Marketing | Classification Evaluation Dashboard
# Evaluation only: saved models are loaded; no retraining.
# ============================================================

st.set_page_config(
    page_title="Bank Marketing | ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

TARGET = "deposit"

LABEL_MAP = {
    "no": 0,
    "yes": 1,
}

CLASS_NAMES = [
    "No subscription",
    "Subscribed",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.035em;
        color: var(--text-color);
        margin-bottom: .15rem;
    }

    .subtitle {
        color: #718096;
        font-size: .98rem;
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        font-size: .72rem;
        letter-spacing: .12em;
        font-weight: 700;
        color: #0f766e;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 12px;
        padding: 12px 14px;
        background: var(--secondary-background-color);
    }

    div[data-testid="stMetric"] label {
        font-size: .8rem;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

left, right = st.columns([5, 1])

with left:

    st.markdown(
        """
        <div class="eyebrow">
            M.Tech AIML · Machine Learning Assignment 2
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-title">'
        'Bank Marketing Model Evaluation'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            Compare five classifiers on labelled test data.
            Explore model performance, errors, and predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:

    st.markdown("")

    st.markdown("**Evaluation mode**")

    st.caption("Models are not retrained")


st.divider()


# ============================================================
# ARTIFACT LOADING
# ============================================================

def resolve_artifact_dir():
    """
    Support artifacts in model/ or beside app.py.
    """

    if (MODEL_DIR / "preprocessor.joblib").is_file():
        return MODEL_DIR

    if (ROOT / "preprocessor.joblib").is_file():
        return ROOT

    return MODEL_DIR


@st.cache_resource(show_spinner="Loading trained models…")
def load_artifacts(artifact_dir_string):

    artifact_dir = Path(artifact_dir_string)

    required = [
        "preprocessor.joblib",
        *MODEL_FILES.values(),
    ]

    missing = [
        file
        for file in required
        if not (artifact_dir / file).is_file()
    ]

    if missing:

        raise FileNotFoundError(
            f"Missing files in {artifact_dir}: "
            f"{', '.join(missing)}"
        )

    preprocessor = joblib.load(
        artifact_dir / "preprocessor.joblib"
    )

    models = {
        name: joblib.load(artifact_dir / filename)
        for name, filename in MODEL_FILES.items()
    }

    feature_names = list(
        getattr(
            preprocessor,
            "feature_names_in_",
            [],
        )
    )

    if not feature_names:

        raise ValueError(
            "The fitted preprocessor has no feature_names_in_. "
            "Check that it is the fitted preprocessing artifact."
        )

    return preprocessor, models, feature_names


artifact_dir = resolve_artifact_dir()

try:

    preprocessor, models, expected_features = load_artifacts(
        str(artifact_dir)
    )

except Exception as exc:

    st.error("The saved ML artifacts could not be loaded.")

    st.code(str(exc))

    st.info(
        "Expected preprocessor.joblib and the five model files "
        "inside a model/ folder beside app.py, "
        "or directly beside app.py."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Evaluation setup")

    st.caption(
        "Bank Marketing · Binary classification"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload labelled test data",
        type=["csv"],
        help=(
            f"CSV must include {len(expected_features)} "
            f"input features and `{TARGET}`."
        ),
    )

    st.markdown("**Required target**")

    st.code(
        "deposit → yes / no",
        language="text",
    )

    st.divider()

    st.markdown("**Loaded artifacts**")

    st.caption(
        f"Preprocessor: `{artifact_dir.name}/preprocessor.joblib`"
    )

    for model_name in MODEL_FILES:

        st.caption(
            f"✓ {model_name}"
        )

    st.divider()

    st.caption(
        "The app transforms uploaded data and evaluates "
        "saved models. It does not fit or retrain."
    )


# ============================================================
# DASHBOARD TABS
# IMPORTANT: Create tabs BEFORE using them.
# ============================================================

overview_tab, comparison_tab, detail_tab, prediction_tab, guide_tab = st.tabs(
    [
        "Overview",
        "Model comparison",
        "Model details",
        "Prediction review",
        "Metric guide",
    ]
)


# ============================================================
# EMPTY STATE
# Dashboard is visible even before uploading a CSV.
# ============================================================

if uploaded_file is None:

    with overview_tab:

        st.subheader("Your evaluation workspace")

        st.info(
            "Upload `test_data.csv` from the sidebar "
            "to populate this dashboard."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Classifiers",
            "5",
        )

        col2.metric(
            "Evaluation metrics",
            "6",
        )

        col3.metric(
            "Evaluation data",
            "Not loaded",
        )

        st.markdown("#### What you can explore")

        a, b, c = st.columns(3)

        with a:

            st.markdown("**Model comparison**")

            st.caption(
                "Compare Accuracy, AUC, Precision, "
                "Recall, F1, and MCC."
            )

        with b:

            st.markdown("**Model diagnostics**")

            st.caption(
                "Inspect confusion matrices and "
                "class-wise classification reports."
            )

        with c:

            st.markdown("**Prediction review**")

            st.caption(
                "Find incorrect predictions and "
                "export results."
            )

    with comparison_tab:

        st.subheader("Model comparison")

        st.info(
            "Upload labelled test data to calculate "
            "and compare the five classifiers."
        )

    with detail_tab:

        st.subheader("Model details")

        st.info(
            "Upload labelled test data to view metrics, "
            "confusion matrix, and classification report."
        )

    with prediction_tab:

        st.subheader("Prediction review")

        st.info(
            "Upload labelled test data to inspect "
            "model predictions."
        )

    with guide_tab:

        st.subheader("Evaluation metrics")

        st.markdown(
            """
            | Metric | What it measures |
            |---|---|
            | Accuracy | Overall fraction of correct predictions |
            | AUC | Ranking ability across classification thresholds |
            | Precision | Reliability of positive predictions |
            | Recall | Fraction of actual positive cases identified |
            | F1-score | Harmonic mean of precision and recall |
            | MCC | Balanced summary using all confusion-matrix outcomes |
            """
        )

    st.stop()


# ============================================================
# READ AND VALIDATE TEST DATA
# ============================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as exc:

    st.error(
        f"Could not read the CSV: {exc}"
    )

    st.stop()


if data.empty:

    st.error(
        "The uploaded CSV contains no rows."
    )

    st.stop()


if TARGET not in data.columns:

    st.error(
        f"Required target column `{TARGET}` was not found."
    )

    st.stop()


missing_features = [
    col
    for col in expected_features
    if col not in data.columns
]

if missing_features:

    st.error(
        "The uploaded CSV is missing these input features:"
    )

    st.write(
        ", ".join(
            f"`{col}`"
            for col in missing_features
        )
    )

    st.stop()


raw_target = (
    data[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
)


unexpected_labels = sorted(
    set(raw_target.unique()) - set(LABEL_MAP)
)

if unexpected_labels:

    st.error(
        f"`{TARGET}` must contain only `yes` and `no`. "
        f"Found: {unexpected_labels}"
    )

    st.stop()


y_true = (
    raw_target
    .map(LABEL_MAP)
    .astype(int)
)

X_raw = data[expected_features].copy()


try:

    with st.spinner("Applying saved preprocessing…"):

        X_test = preprocessor.transform(X_raw)

except Exception as exc:

    st.error(
        f"Preprocessing failed: {exc}"
    )

    st.stop()


if y_true.nunique() < 2:

    st.warning(
        "The uploaded test file contains only one class. "
        "AUC will be unavailable."
    )


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def positive_scores(model, X):

    """
    Return the score associated with positive class 1.
    """

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        classes = list(
            getattr(
                model,
                "classes_",
                [0, 1],
            )
        )

        if 1 in classes:

            return probabilities[:, classes.index(1)]

        return None

    if hasattr(model, "decision_function"):

        scores = model.decision_function(X)

        classes = list(
            getattr(
                model,
                "classes_",
                [0, 1],
            )
        )

        if len(classes) == 2 and classes[1] == 1:

            return scores

        if len(classes) == 2 and classes[0] == 1:

            return -scores

    return None


def evaluate(name, model, X, y):

    model_input = X

    # GaussianNB requires dense input.
    if (
        name == "Gaussian Naive Bayes"
        and hasattr(X, "toarray")
    ):

        model_input = X.toarray()

    y_pred = model.predict(model_input)

    score = positive_scores(
        model,
        model_input,
    )

    auc = np.nan

    if score is not None and y.nunique() == 2:

        try:

            auc = roc_auc_score(
                y,
                score,
            )

        except (ValueError, TypeError):

            pass

    metrics = {

        "Accuracy": accuracy_score(
            y,
            y_pred,
        ),

        "AUC": auc,

        "Precision": precision_score(
            y,
            y_pred,
            zero_division=0,
        ),

        "Recall": recall_score(
            y,
            y_pred,
            zero_division=0,
        ),

        "F1-score": f1_score(
            y,
            y_pred,
            zero_division=0,
        ),

        "MCC": matthews_corrcoef(
            y,
            y_pred,
        ),
    }

    return metrics, y_pred


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

results = {}

predictions = {}

evaluation_errors = {}


with st.spinner("Evaluating five saved classifiers…"):

    for name, model in models.items():

        try:

            metrics, y_pred = evaluate(
                name,
                model,
                X_test,
                y_true,
            )

            results[name] = metrics

            predictions[name] = y_pred

        except Exception as exc:

            evaluation_errors[name] = str(exc)


if evaluation_errors:

    with st.expander(
        "View model evaluation warnings"
    ):

        for name, message in evaluation_errors.items():

            st.warning(
                f"{name}: {message}"
            )


if not results:

    st.error(
        "No models could be evaluated. "
        "Check that saved models match the preprocessor."
    )

    st.stop()


comparison = pd.DataFrame.from_dict(
    results,
    orient="index",
)

comparison.index.name = "Model"

comparison = (
    comparison
    .reset_index()
    .sort_values(
        "MCC",
        ascending=False,
        na_position="last",
    )
)

best = comparison.iloc[0]


# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:

    st.subheader("Evaluation overview")

    st.caption(
        "Results below are calculated from "
        "the uploaded labelled test dataset."
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Test records",
        f"{len(data):,}",
    )

    m2.metric(
        "Input features",
        f"{len(expected_features)}",
    )

    m3.metric(
        "Models evaluated",
        f"{len(results)} / 5",
    )

    m4.metric(
        "Top model by MCC",
        best["Model"],
        f"{best['MCC']:.4f}",
    )

    left, right = st.columns(
        [1.2, 1]
    )

    with left:

        st.markdown(
            "#### Test data preview"
        )

        st.dataframe(
            data.head(8),
            width="stretch",
            hide_index=True,
        )

    with right:

        st.markdown(
            "#### Target distribution"
        )

        counts = (
            raw_target
            .value_counts()
            .reindex(
                ["no", "yes"],
                fill_value=0,
            )
        )

        fig, ax = plt.subplots(
            figsize=(5, 3.5)
        )

        bars = ax.bar(
            ["No", "Yes"],
            counts.values,
        )

        ax.set_ylabel(
            "Number of records"
        )

        ax.spines[
            ["top", "right"]
        ].set_visible(False)

        ax.bar_label(
            bars,
            padding=3,
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            width="stretch",
        )

        plt.close(fig)

    st.success(
        f"**{best['Model']}** currently ranks highest "
        "by MCC on this uploaded test set. "
        "The ranking may change with a different test dataset."
    )


# ============================================================
# MODEL COMPARISON TAB
# ============================================================

with comparison_tab:

    st.subheader(
        "Compare classifiers"
    )

    st.caption(
        "All five models are evaluated on "
        "the same uploaded test data."
    )

    shown = comparison.copy()

    for metric in [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1-score",
        "MCC",
    ]:

        shown[metric] = shown[metric].map(
            lambda value:
                "—"
                if pd.isna(value)
                else f"{value:.4f}"
        )

    st.dataframe(
        shown,
        width="stretch",
        hide_index=True,
    )

    metric_choice = st.selectbox(
        "Metric to visualize",
        [
            "MCC",
            "F1-score",
            "AUC",
            "Accuracy",
            "Precision",
            "Recall",
        ],
    )

    chart = (
        comparison[
            [
                "Model",
                metric_choice,
            ]
        ]
        .set_index("Model")
    )

    st.bar_chart(
        chart,
        width="stretch",
    )

    st.download_button(
        "Download metrics as CSV",
        data=comparison.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="model_comparison_metrics.csv",
        mime="text/csv",
    )


# ============================================================
# MODEL DETAILS TAB
# ============================================================

with detail_tab:

    st.subheader(
        "Model diagnostics"
    )

    selected_model = st.selectbox(
        "Select classifier",
        list(results.keys()),
        key="detail_model",
    )

    selected_metrics = results[
        selected_model
    ]

    metric_cols = st.columns(6)

    for col, metric in zip(
        metric_cols,
        [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1-score",
            "MCC",
        ],
    ):

        value = selected_metrics[
            metric
        ]

        col.metric(
            metric,
            "—"
            if pd.isna(value)
            else f"{value:.4f}",
        )

    y_pred = predictions[
        selected_model
    ]

    cm_col, report_col = st.columns(2)

    with cm_col:

        st.markdown(
            "#### Confusion matrix"
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        fig, ax = plt.subplots(
            figsize=(5.4, 4.2)
        )

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASS_NAMES,
        ).plot(
            ax=ax,
            values_format="d",
            colorbar=False,
            cmap="Blues",
        )

        ax.set_title(
            selected_model
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            width="stretch",
        )

        plt.close(fig)

    with report_col:

        st.markdown(
            "#### Classification report"
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        st.dataframe(
            pd.DataFrame(
                report
            ).T.round(4),
            width="stretch",
        )


# ============================================================
# PREDICTION REVIEW TAB
# ============================================================

with prediction_tab:

    st.subheader(
        "Prediction review"
    )

    review_model = st.selectbox(
        "Select classifier",
        list(results.keys()),
        key="prediction_model",
    )

    y_pred = predictions[
        review_model
    ]

    review = data.copy()

    review["Actual deposit"] = (
        y_true
        .map(
            {
                0: "no",
                1: "yes",
            }
        )
        .to_numpy()
    )

    review["Predicted deposit"] = (
        pd.Series(
            y_pred,
            index=data.index,
        )
        .map(
            {
                0: "no",
                1: "yes",
            }
        )
    )

    review["Correct prediction"] = (
        review["Actual deposit"]
        == review["Predicted deposit"]
    )

    errors_only = st.checkbox(
        "Show incorrect predictions only",
        value=False,
    )

    view = (
        review.loc[
            ~review["Correct prediction"]
        ]
        if errors_only
        else review
    )

    st.caption(
        f"Showing up to 500 rows "
        f"from {len(view):,} matching records."
    )

    st.dataframe(
        view.head(500),
        width="stretch" ,
        hide_index=True,
    )

    st.download_button(
        "Download prediction results",
        data=review.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            f"{review_model.lower().replace(' ', '_')}"
            "_predictions.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# METRIC GUIDE TAB
# ============================================================

with guide_tab:

    st.subheader(
        "Understanding the evaluation metrics"
    )

    st.markdown(
        """
        | Metric | Interpretation |
        |---|---|
        | **Accuracy** | Fraction of all predictions that are correct. |
        | **AUC** | Ability to rank positive cases above negative cases across thresholds. |
        | **Precision** | Among predicted subscribers, the fraction who actually subscribed. |
        | **Recall** | Among actual subscribers, the fraction identified by the model. |
        | **F1-score** | Harmonic mean of precision and recall. |
        | **MCC** | Balanced summary using TP, TN, FP, and FN; ranges from −1 to +1. |
        """
    )

    st.info(
        "AUC is unavailable when the uploaded test data "
        "contains only one target class."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "M.Tech AIML · Machine Learning Assignment 2 "
    "· Bank Marketing classification"
)