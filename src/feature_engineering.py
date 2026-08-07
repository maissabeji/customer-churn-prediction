"""
Feature engineering pipeline for the Telco Customer Churn dataset.
Encodes categorical features and drops redundant/non-feature columns,
based on decisions documented in notebooks/eda_findings.md.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DROP_COLS = ['customerID', 'TotalCharges']
TARGET_COL = 'Churn'

BINARY_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
ONEHOT_COLS = [
    'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaymentMethod'
]
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'SeniorCitizen']


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw data and apply only the cleaning steps validated in Milestone 2."""
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df


def prepare_xy(df: pd.DataFrame):
    """Split into features (X) and binary-encoded target (y). Drops non-feature columns."""
    y = (df[TARGET_COL] == 'Yes').astype(int)
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    for col in BINARY_COLS:
        X[col] = (X[col] == 'Yes').astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Returns an UNFITTED ColumnTransformer.
    Must be .fit() on TRAINING data only (Milestone 5) — never on full dataset,
    to avoid leaking test-set category distributions into training.
    """
    return ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ONEHOT_COLS),
        ],
        remainder='passthrough'  # numeric + already-binary-encoded columns pass through unchanged
    )


if __name__ == "__main__":
    df = load_and_clean("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X, y = prepare_xy(df)
    print("X shape:", X.shape)
    print("X columns:", X.columns.tolist())
    print("y balance:\n", y.value_counts(normalize=True))