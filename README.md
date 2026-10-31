# Machine Learning Assignment 2

A Streamlit-based machine learning application for evaluating and comparing classification models on a dataset.
## Problem Statement

The objective of this project is to implement and compare five machine learning classification algorithms on the Bank Marketing dataset. The models predict whether a client will subscribe to a term deposit (`yes` or `no`). Their performance is evaluated using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

## Dataset Description

- **Dataset:** Bank Marketing
- **Task:** Binary classification
- **Target variable:** `deposit`
- **Target classes:** `yes` and `no`
- **Test records evaluated:** 2,233
- **Input features:** 16
- **Dataset files:** `bank.csv` and `test_data.csv`

The application uses the labelled test dataset to evaluate the saved models. The models are loaded from the `model/` directory and are not retrained during evaluation.

## GitHub Repository

[View the complete source code and model artifacts](https://github.com/kunjal2004/ML_Assignment)

## Live Streamlit Application

[Open the deployed Machine Learning Assignment 2 application](https://mlassignment-fqsw3zzjvkf3t5lqsr2rur.streamlit.app/)

## Overview

This project provides an interactive interface to:

- Upload a test dataset in CSV format
- Evaluate multiple trained classification models
- Compare model performance using common evaluation metrics
- View confusion matrices and classification reports
- Review predictions
- Download evaluation results

## Models Implemented

The application supports the following classification models:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The application also displays classification reports to provide a more detailed view of each model’s performance.

## Project Structure

```text
assignment_2/
├── app.py
├── model/
│   ├── preprocessor.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── test_data.csv
├── requirements.txt
└── README.md
```

> **Note:** The trained model files and test dataset should be placed in the locations expected by `app.py`. If the model artifacts are stored outside the `model/` directory, update the artifact path in the application accordingly.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kunjal2004/ML_Assignment.git
cd ML_Assignment
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib joblib
```

## Run the Application

From the project directory, execute:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

## How to Use

1. Start the Streamlit application.
2. Upload a test dataset in CSV format using the sidebar.
3. Ensure the dataset contains the expected target column, `deposit`.
4. The application evaluates the available trained models.
5. Navigate through the tabs to explore:
   - **Overview:** Dataset and evaluation summary
   - **Model Comparison:** Performance comparison across models
   - **Model Details:** Individual model metrics and reports
   - **Prediction Review:** Review model predictions
   - **Metric Guide:** Explanation of evaluation metrics
6. Download available evaluation results if required.

## Dataset Requirements

The uploaded CSV should:

- Contain the target column expected by the application
- Include the feature columns used during model training
- Use compatible data types and values
- Be prepared using the same preprocessing pipeline as the trained models

The current application is configured to use `deposit` as the target column. Update the `TARGET` variable in `app.py` if your assignment uses a different target.

## Model Artifacts

The application loads serialized models and the preprocessing pipeline using Joblib.

Expected files:

| File | Description |
|---|---|
| `preprocessor.joblib` | Preprocessing pipeline |
| `logistic_regression.joblib` | Trained Logistic Regression model |
| `decision_tree.joblib` | Trained Decision Tree model |
| `knn.joblib` | Trained KNN model |
| `naive_bayes.joblib` | Trained Naive Bayes model |
| `svm.joblib` | Trained SVM model |

Ensure these files are available before running the application.

## Troubleshooting

### App does not start

Check that the virtual environment is activated and dependencies are installed.

```bash
pip install -r requirements.txt
```

### Model files are not found

Verify that all `.joblib` files exist in the expected artifact directory and that the paths in `app.py` match the project structure.

### Uploaded dataset is rejected

Check that the CSV contains the expected target column and the required feature columns.

### Streamlit displays warnings

Use the current Streamlit layout parameters in `app.py`, such as:

```python
st.dataframe(data, width="stretch")
```

## Author

Kunjal Chachra
