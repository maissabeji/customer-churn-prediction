"""
Model training pipeline for customer churn prediction.
Trains a RandomForest classifier on the Telco dataset and evaluates on held-out test set.
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from src.feature_engineering import load_and_clean, prepare_xy, build_preprocessor


def train():
    # 1. Load and clean
    df = load_and_clean("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # 2. Prepare features and target
    X, y = prepare_xy(df)

    # 3. Split — stratify preserves 73.5/26.5 class ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Build pipeline: preprocessor → classifier
    prep = build_preprocessor()
    pipe = Pipeline(steps=[
        ('preprocessor', prep),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 5. Fit on training data ONLY
    pipe.fit(X_train, y_train)

    # 6. Evaluate on held-out test set
    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

    return pipe


if __name__ == "__main__":
    train()