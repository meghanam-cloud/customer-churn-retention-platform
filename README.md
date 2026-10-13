# Customer Churn & Retention Intelligence Platform

An end-to-end machine learning project that analyzes customer churn patterns, predicts churn risk, and lays the foundation for data-driven customer retention.

## Project Overview

Customer churn is a key challenge for subscription-based businesses. This project uses the IBM Telco Customer Churn sample dataset to explore churn behavior, compare classification models, and develop a CatBoost model to predict whether a customer is likely to leave.

The project also uses MLflow to track model parameters, evaluation metrics, and the saved model artifact.

## Objectives

- Analyze customer churn patterns through exploratory data analysis.
- Prepare and preprocess numerical and categorical features.
- Compare multiple machine learning algorithms.
- Tune a CatBoost classifier and select a decision threshold using validation data.
- Evaluate the selected model on held-out test data.
- Track model results and artifacts with MLflow.
- Prepare the project for deployment and customer-retention applications.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- CatBoost
- Matplotlib and Seaborn
- MLflow
- Jupyter Notebook
- Git and GitHub

## Current Model Performance

The selected CatBoost model was evaluated on the held-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 77.08% |
| Precision | 55.63% |
| Recall | 67.38% |
| F1 Score | 60.94% |
| ROC-AUC | 83.64% |

The decision threshold of 0.60 was selected using validation data. The results reflect a trade-off between identifying customers who churn and avoiding false-positive predictions.

These results are based on the IBM Telco Customer Churn sample dataset, not live customer data.

## Repository Structure

```text
customer-churn-retention-platform/
├── app/
│   └── main.py
├── models/
│   ├── churn_catboost.cbm
│   └── churn_metadata.json
├── notebooks/
│   └── 01_data_understanding.ipynb
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/meghanam-cloud/customer-churn-retention-platform.git
cd customer-churn-retention-platform
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages required by the notebook and application. A complete `requirements.txt` will be added as the project is finalized.

### 4. Obtain the dataset

Use the IBM Telco Customer Churn sample dataset. Place `Telco-Customer-Churn.csv` inside the `data/` directory. The raw CSV is intentionally excluded from this repository.

### 5. Explore the notebook

Open `notebooks/01_data_understanding.ipynb` in Jupyter Notebook or VS Code and run the cells in order.

## Experiment Tracking

The selected CatBoost model and its parameters and evaluation metrics have been logged in a local MLflow experiment named `Customer-Churn-Retention`.
      
The project will be extended with Databricks-based experiment tracking, model management, and a prediction interface.

## Current Status

- Data cleaning and exploratory analysis completed.
- Multiple model families evaluated.
- CatBoost model tuned and evaluated.
- Model artifact and metadata saved.
- Initial MLflow experiment logged.
- Project uploaded to GitHub.

## Limitations

- The dataset is a sample dataset and may not represent real-world customer behavior.
- Predictive associations do not establish the causes of churn.
- Model performance and decision thresholds may change on different datasets.
- A production deployment and end-user prediction interface are still in progress.

## Future Improvements

- Integrate Databricks and managed model tracking.
- Add a prediction API and interactive Streamlit interface.
- Add automated tests and input validation.
- Document reproducible installation and deployment steps.

## Author

Built as a portfolio project to demonstrate data analysis, machine learning, model evaluation, and experiment tracking.