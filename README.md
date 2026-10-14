# Customer Churn & Retention Intelligence Platform

An end-to-end machine learning project that analyzes customer churn patterns, compares classification models, and predicts customer churn risk through an API and interactive web interface.

## Project Overview

Customer churn is an important challenge for subscription-based businesses. This project uses the IBM Telco Customer Churn sample dataset to explore churn patterns, evaluate machine learning algorithms, and develop a CatBoost classification model.

The selected model is exposed through a FastAPI backend and a Streamlit interface. MLflow is used for experiment tracking, and the trained model has also been logged and registered in Databricks Unity Catalog.

**Dataset note:** Results are based on a sample dataset and do not represent a live customer deployment.

## Project Objectives

- Perform data cleaning and exploratory data analysis.
- Investigate relationships between customer characteristics and churn.
- Compare multiple classification algorithms.
- Tune a CatBoost model and select a decision threshold using validation data.
- Evaluate model performance on held-out test data.
- Track model experiments with MLflow.
- Register the model in Databricks Unity Catalog.
- Serve predictions through FastAPI and Streamlit.

## Technology Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn, CatBoost
- **Visualization:** Matplotlib, Seaborn
- **Experiment tracking:** MLflow
- **Model management:** Databricks Unity Catalog
- **API:** FastAPI, Uvicorn
- **Interactive UI:** Streamlit
- **Development:** Jupyter Notebook, Git, GitHub

## Model Evaluation

The selected CatBoost model was evaluated on the held-out test set using a decision threshold of 0.60, selected using validation data.

| Metric | Test Result |
|---|---:|
| Accuracy | 77.08% |
| Precision | 55.63% |
| Recall | 67.38% |
| F1 Score | 60.94% |
| ROC-AUC | 83.64% |

The threshold determines how predicted churn probabilities are converted into Yes/No predictions. A threshold of 0.60 means probabilities at or above 0.60 are classified as churn.

These metrics reflect the evaluation on this sample dataset; performance may differ on other customer populations.

## Repository Structure

```text
customer-churn-retention-platform/
├── app/
│   ├── main.py
│   └── streamlit_app.py
├── models/
│   ├── churn_catboost.cbm
│   └── churn_metadata.json
├── notebooks/
│   └── 01_data_understanding.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The raw dataset is excluded from the repository and must be obtained separately.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/meghanam-cloud/customer-churn-retention-platform.git
cd customer-churn-retention-platform
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the FastAPI backend

Run in the first terminal:

```powershell
python -m uvicorn app.main:app --reload
```

Useful endpoints:

- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Prediction endpoint: `POST /predict`

Keep the backend running while using the interface.

### 5. Start the Streamlit interface

Open a second terminal in the project directory, activate the same virtual environment, and run:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL displayed in the terminal, usually http://localhost:8501.

Enter customer details in the interface to request a churn prediction from the FastAPI backend.

### 6. Explore the notebook

Obtain the IBM Telco Customer Churn sample dataset and place `Telco-Customer-Churn.csv` inside the `data/` directory.

Open `notebooks/01_data_understanding.ipynb` in Jupyter Notebook or VS Code to review the data preparation and exploratory analysis.

## MLflow and Databricks

The project uses MLflow to track experiment parameters, evaluation metrics, and model artifacts.

The model was also logged in Databricks and registered in Unity Catalog as:

`ai_projects_catalog.default.customer_churn_model`

The registered model is available for model management and further deployment workflows. A managed model-serving endpoint is not part of the current local prediction setup.

## Limitations

- The dataset is a sample and may not represent real-world customer behavior.
- Associations between features and churn do not establish causation.
- The chosen decision threshold reflects this project's validation approach and business trade-offs.
- The local FastAPI and Streamlit applications are not a production-hosted service.
- Model performance should be re-evaluated before using the model with real customer data.

## Future Improvements

- Add automated unit and API tests.
- Introduce data and prediction monitoring.
- Evaluate model calibration and threshold choices against explicit retention costs.
- Add a reproducible training pipeline and model versioning workflow.
- Extend the analysis with SQL-based customer analytics and a business intelligence dashboard.

## Author

Built as a portfolio project demonstrating data analysis, classification modeling, model evaluation, experiment tracking, model management, API development, and interactive machine learning applications.