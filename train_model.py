"""
Train the churn prediction model on the Telco Customer Churn dataset
and save the fitted model + encoders for use in the Streamlit app.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract',
            'PaymentMethod', 'InternetService', 'OnlineSecurity',
            'TechSupport', 'PaperlessBilling', 'SeniorCitizen']

def load_and_clean(path="telco.csv"):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    df['Churn_Flag'] = (df['Churn'] == 'Yes').astype(int)
    return df

def train():
    df = load_and_clean()
    X = df[FEATURES].copy()
    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    y = df['Churn_Flag']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"AUC: {roc_auc_score(y_test, proba):.4f}")

    joblib.dump({"model": model, "encoders": encoders, "features": FEATURES}, "churn_model.pkl")
    print("Saved churn_model.pkl")

if __name__ == "__main__":
    train()
