# Machine Learning Assignment 2
## Bank Marketing Classification and Model Evaluation

A Streamlit-based machine learning application for evaluating and comparing five classification models on the Bank Marketing dataset.

## 1. Problem Statement

The objective of this project is to implement and compare five machine learning classification algorithms using the Bank Marketing dataset. The models predict whether a client will subscribe to a term deposit (`yes` or `no`) based on customer and campaign-related attributes.

The models are evaluated using Accuracy, ROC-AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

- **Dataset:** Bank Marketing
- **Task:** Binary classification
- **Target variable:** `deposit`
- **Target classes:** `yes` and `no`
- **Input features:** 16
- **Test records evaluated:** 2,233
- **Dataset files:** `bank.csv` and `test_data.csv`

The application uses labelled test data to evaluate the saved models. The trained models and preprocessing pipeline are loaded from the `model/` directory. The application does not retrain the models during evaluation.

## 3. GitHub Repository

The repository contains the source code, trained models, preprocessing artifacts, test data, dependencies, and documentation.

[View the GitHub Repository](https://github.com/kunjal2004/ML_Assignment)

## 4. Live Streamlit Application

The deployed application allows users to upload a labelled CSV dataset, select a trained model, and view its evaluation results.

[Open the Live Streamlit Application](https://mlassignment-fqsw3zzjvkf3t5lqsr2rur.streamlit.app/)

## 5. Project Overview

The application provides an interactive interface to:

- Upload a test dataset in CSV format
- Evaluate trained classification models
- Compare model performance using common evaluation metrics
- View confusion matrices and classification reports
- Review model predictions
- Download evaluation results, where available

## 6. Models Implemented

The following classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 7. Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Proportion of correctly classified records
- **ROC-AUC:** Measures the model’s ability to distinguish between the two classes across classification thresholds
- **Precision:** Proportion of predicted positive records that are actually positive
- **Recall:** Proportion of actual positive records correctly identified
- **F1-score:** Harmonic mean of Precision and Recall
- **MCC:** Correlation between actual and predicted classes, taking all confusion-matrix outcomes into account

## 8. Model Comparison Results

The following results were obtained on the labelled test dataset containing 2,233 records.

| ML Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8262 | 0.9071 | 0.8278 | 0.7996 | 0.8135 | 0.6513 |
| Decision Tree | 0.8186 | 0.8624 | 0.7822 | 0.8554 | 0.8172 | 0.6405 |
| K-Nearest Neighbors | 0.8173 | 0.8796 | 0.8199 | 0.7873 | 0.8033 | 0.6333 |
| Gaussian Naive Bayes | 0.7201 | 0.8042 | 0.7837 | 0.5652 | 0.6568 | 0.4472 |
| Support Vector Machine | **0.8554** | **0.9194** | 0.8249 | **0.8819** | **0.8524** | **0.7124** |

## 9. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved an accuracy of 82.62% and an AUC of 0.9071, indicating good class discrimination. Its precision was 0.8278 and recall was 0.7996, showing a reasonably balanced performance. It provides a strong baseline for this classification task. |
| **Decision Tree** | The Decision Tree achieved a recall of 0.8554, higher than Logistic Regression and KNN, meaning it identified a relatively large proportion of actual positive cases. Its precision was lower at 0.7822, indicating more false-positive predictions. Its overall MCC was 0.6405. |
| **K-Nearest Neighbors (KNN)** | KNN achieved an accuracy of 81.73% and an AUC of 0.8796. Its precision of 0.8199 was relatively strong, but its recall of 0.7873 was lower than that of the Decision Tree and SVM. Its overall performance was competitive but below SVM. |
| **Gaussian Naive Bayes** | Gaussian Naive Bayes achieved the lowest accuracy, F1-score, recall, and MCC among the five models. Its recall of 0.5652 indicates that it missed a comparatively larger proportion of actual positive cases. It had the weakest overall performance on this test dataset. |
| **Support Vector Machine (SVM)** | SVM achieved the highest accuracy (0.8554), AUC (0.9194), recall (0.8819), F1-score (0.8524), and MCC (0.7124). Its results indicate the strongest overall performance and the best balance of classification quality among the evaluated models. |
| **Overall winner** | **Support Vector Machine (SVM)** is the overall winner for this dataset based on the evaluated test results. It achieved the highest MCC and the strongest overall combination of accuracy, AUC, recall, and F1-score. |

## 10. Project Structure

```text
ML_Assignment/
├── app.py
├── ML_models.ipynb
├── model/
│   ├── preprocessor.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── bank.csv
├── test_data.csv
├── requirements.txt
└── README.md
```

## 11. Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 12. Installation

### Clone the repository

```bash
git clone https://github.com/kunjal2004/ML_Assignment.git
cd ML_Assignment
```

### Create and activate a virtual environment

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

### Install dependencies

```bash
pip install -r requirements.txt
```

## 13. Run the Application

From the project directory, execute:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in a browser to use the application.

## 14. How to Use

1. Start the Streamlit application.
2. Upload a labelled test dataset in CSV format using the sidebar.
3. Ensure the dataset contains the expected target column, `deposit`.
4. Select a trained model for evaluation.
5. Explore the available application sections:
   - **Overview:** Dataset and evaluation summary
   - **Model Comparison:** Performance comparison across models
   - **Model Details:** Individual model metrics and reports
   - **Prediction Review:** Review model predictions
   - **Metric Guide:** Explanation of evaluation metrics
6. Download evaluation results if the option is available.

## 15. Dataset Requirements

The uploaded CSV should:

- Contain the target column `deposit`
- Include the feature columns used during model training
- Use compatible data types and values
- Be compatible with the saved preprocessing pipeline

The application is configured to use `deposit` as the target variable.

## 16. Model Artifacts

The application loads serialized models and the preprocessing pipeline using Joblib.

| File | Description |
|---|---|
| `preprocessor.joblib` | Saved preprocessing pipeline |
| `logistic_regression.joblib` | Trained Logistic Regression model |
| `decision_tree.joblib` | Trained Decision Tree model |
| `knn.joblib` | Trained KNN model |
| `naive_bayes.joblib` | Trained Gaussian Naive Bayes model |
| `svm.joblib` | Trained Support Vector Machine model |

These artifacts must be available in the expected directory before the application is run.

## 17. Troubleshooting

### Application does not start

Check that the virtual environment is activated and dependencies are installed.

```bash
pip install -r requirements.txt
```

### Model files are not found

Verify that all `.joblib` files exist in the expected artifact directory and that the paths in `app.py` match the project structure.

### Uploaded dataset is rejected

Check that the CSV contains the expected target column and all required feature columns.

## 18. Conclusion

This project implemented and compared five classification algorithms using the Bank Marketing dataset to predict term-deposit subscription. The models were evaluated using Accuracy, ROC-AUC, Precision, Recall, F1-score, and MCC.

Based on the evaluated test results, Support Vector Machine achieved the strongest overall performance, with an accuracy of 0.8554, AUC of 0.9194, and MCC of 0.7124. The project also includes execution in BITS Virtual Lab, source-code version control through GitHub, and deployment of an interactive Streamlit application.

## Author

Kunjal Chachra
