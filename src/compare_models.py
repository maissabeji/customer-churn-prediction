from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier

from src.feature_engineering import load_and_clean, prepare_xy, build_preprocessor


def compare_models():
    mlflow.set_experiment("churn-model-comparison")
        # Hyperparameters — defined as variables so they're logged and used consistently
    n_estimators = 100
    random_state = 42
    test_size = 0.2
    df = load_and_clean("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # 2. Prepare features and target
    X, y = prepare_xy(df)

    # 3. Split — stratify preserves 73.5/26.5 class ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    with mlflow.start_run(run_name="random-forest-v2"):    
        

        mlflow.log_params({
            "n_estimators": n_estimators,
            "random_state": random_state,
            "test_size": test_size
        })

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


        # 7. Log fitted pipeline as MLflow artifact
        mlflow.sklearn.log_model(pipe, name="model")

    with mlflow.start_run(run_name="xgboost-baseline"):
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        mlflow.log_params({
            "scale_pos_weight": round(float(scale_pos_weight), 2),
            "random_state": random_state,
            "test_size": test_size
        })
        prep_xgb = build_preprocessor()
        pipe_xgb =  Pipeline(steps=[
            ('preprocessor', prep_xgb),
            ('classifier', XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                eval_metric='logloss',
                verbosity=0
            ))
        ])
        pipe_xgb.fit(X_train,y_train)

        y_pred_xgb = pipe_xgb.predict(X_test)
        y_pred_proba_xgb = pipe_xgb.predict_proba(X_test)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_xgb, average=None
        )
        roc_auc = roc_auc_score(y_test, y_pred_proba_xgb)

        mlflow.log_metrics({
            "precision_churn": precision[1],
            "recall_churn": recall[1],
            "f1_churn": f1[1],
            "roc_auc": roc_auc
        })

        mlflow.sklearn.log_model(
            pipe_xgb,
            name="model",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )        
        print("\n--- Threshold Tuning (XGBoost) ---")
        print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 48)
        for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            y_pred_thresh = (y_pred_proba_xgb >= threshold).astype(int)
            p, r, f, _ = precision_recall_fscore_support(
                y_test, y_pred_thresh, average=None, zero_division=0
            )
            print(f"{threshold:<12} {p[1]:<12.3f} {r[1]:<12.3f} {f[1]:<12.3f}")


if __name__ == "__main__":
    compare_models()