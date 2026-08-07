"""
Model training pipeline for customer churn prediction.
Trains a RandomForest classifier on the Telco dataset and evaluates on held-out test set.
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn

from src.feature_engineering import load_and_clean, prepare_xy, build_preprocessor


def train():
    with mlflow.start_run(run_name="baseline-random-forest"):
        # Hyperparameters — defined as variables so they're logged and used consistently
        n_estimators = 100
        random_state = 42
        test_size = 0.2

        mlflow.log_params({
            "n_estimators": n_estimators,
            "random_state": random_state,
            "test_size": test_size
        })

        # 1. Load and clean
        df = load_and_clean("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        # 2. Prepare features and target
        X, y = prepare_xy(df)

        # 3. Split — stratify preserves 73.5/26.5 class ratio in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # 4. Build pipeline: preprocessor → classifier
        prep = build_preprocessor()
        pipe = Pipeline(steps=[
            ('preprocessor', prep),
            ('classifier', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
        ])

        # 5. Fit on training data ONLY
        pipe.fit(X_train, y_train)

        # 6. Evaluate on held-out test set
        y_pred = pipe.predict(X_test)
        y_pred_proba = pipe.predict_proba(X_test)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        mlflow.log_metrics({
            "precision_churn": precision[1],
            "recall_churn": recall[1],
            "f1_churn": f1[1],
            "roc_auc": roc_auc
        })

        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("ROC AUC Score:", roc_auc)

        # 7. Log fitted pipeline as MLflow artifact
        mlflow.sklearn.log_model(pipe, name="model")

        return pipe


if __name__ == "__main__":
    train()