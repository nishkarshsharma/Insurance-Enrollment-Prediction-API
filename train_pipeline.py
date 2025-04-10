# train_pipeline.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv("employee_data.csv")

# Drop ID column
df = df.drop(columns=["employee_id"])

# Split features and target
X = df.drop("enrolled", axis=1)
y = df["enrolled"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define feature types
numeric_features = ["age", "salary", "tenure_years"]
categorical_features = [
    "gender", "marital_status", "employment_type", "region", "has_dependents"
]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Full pipeline with classifier
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Fit model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\n📈 ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Save model
joblib.dump(pipeline, "model.joblib")
print("\n✅ Model saved to model.joblib")
