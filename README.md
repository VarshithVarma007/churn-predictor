# Customer Churn Prediction Tool

Predicts customer churn risk using the public Telco Customer Churn dataset,
with SQL-based feature analysis and a Streamlit interface.

## What it does
- Loads and cleans the Telco Customer Churn dataset (7,032 records after cleaning)
- Runs SQL queries (via SQLite) including an aggregation by contract type and a
  window-function query (RANK) to rank tenure buckets by churn risk
- Trains a Logistic Regression classifier on 10 engineered features
- Serves predictions through a Streamlit app with a churn-probability score and
  top risk drivers per customer

## Real results (held-out test set, 20% split)
- **Accuracy: 79.1%**
- **AUC: 0.832**
- Features used: 10
- Train/test size: 5,625 / 1,407

## Key SQL-derived insight
Churn risk is heavily concentrated in month-to-month contracts (~43% churn)
vs. two-year contracts (~3% churn), and in customers with less than 12 months
of tenure (~49% churn vs. ~12% for 36+ months).

## Run it locally
```bash
pip install streamlit pandas scikit-learn joblib
python train_model.py   # trains and saves churn_model.pkl
streamlit run app.py
```

## Deploy
Push this folder to a GitHub repo, then deploy for free at
https://share.streamlit.io (Streamlit Community Cloud), pointing it at `app.py`.

## Data source
IBM Telco Customer Churn dataset (public), via
https://github.com/IBM/telco-customer-churn-on-icp4d
