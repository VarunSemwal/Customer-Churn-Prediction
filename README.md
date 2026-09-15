# Telco Customer Churn Prediction & Analytics
## End-to-End Machine Learning Pipeline & REST API

An end-to-end Machine Learning solution built to predict customer churn for a telecommunications provider using the IBM Telco Customer Churn dataset. The project includes complete data preprocessing, exploratory data analysis, feature engineering, decision tree modeling, model evaluation, feature importance analysis, and deployment via a FastAPI REST API.

---

## 📁 Repository Structure

```
customer_churn_project/
│
├── data/
│   └── TelcoCustomerChurn.csv          # Raw Telco Customer Churn Dataset (IBM)
│
├── notebook/
│   └── churn_analysis.ipynb            # Fully executed Jupyter Notebook with analysis & visualizations
│
├── model/
│   └── churn_model.pkl                 # Serialized scikit-learn Pipeline (preprocessing + model)
│
├── app.py                              # FastAPI REST API application
├── requirements.txt                    # Project Python dependencies
├── sample_request.json                 # Sample payload for REST API testing
└── README.md                           # Project documentation & instructions
```

---

## 🚀 Quick Start & Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher installed.

### 2. Environment Setup
Clone or navigate to the project workspace and create a Python virtual environment:

```bash
cd customer_churn_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 📓 Running the Analysis Notebook

The Jupyter notebook contains the complete workflow including Data Understanding, Cleaning, 5 Business EDA Visualizations, Feature Engineering, Model Training, Evaluation, and Interpretation.

To launch Jupyter Notebook:

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

---

## 🌐 Running the FastAPI REST API Server

Start the local API web server using Uvicorn:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive OpenAPI (Swagger) Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📡 Sample API Request & Response

### Endpoint: `POST /predict`

#### Sample Request Payload (High Risk Customer)

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.70,
  "TotalCharges": 151.65
}
```

#### Sample Response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.85
}
```

#### Testing with cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_request.json
```

#### Testing with Python `requests`

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.70,
    "TotalCharges": 151.65
}

response = requests.post(url, json=payload)
print(response.json())
# Output: {'prediction': 'Yes', 'churn_probability': 0.85}
```

---

## 📊 Summary of Model Findings & Business Business Rationale

1. **Model Selection**:
   - Evaluated 3 Decision Tree configurations.
   - Selected **Config 2 (Regularized Gini, Depth 5, Balanced Weights)** which achieved **79.8% Recall on Churn**.
2. **Precision vs. Recall Priority**:
   - In telecom retention, **Recall is prioritized over Precision**. Missing an actual churner results in lost Customer Lifetime Value (LTV), whereas sending an unnecessary discount to a non-churning customer incurs a minimal marginal cost.
3. **Key Drivers**:
   - Contract type (Month-to-month), Fiber Optic internet service without Tech Support/Online Security, and tenure < 12 months are the top predictors of churn risk.
