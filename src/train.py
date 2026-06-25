import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

EVAL_THRESHOLD = 0.70


def _build_model(params: dict):
    model_type = params.get("model_type", "random_forest")
    tree_params = {k: v for k, v in params.items() if k not in ("model_type",)}

    if model_type == "gradient_boosting":
        allowed = {"n_estimators", "max_depth", "min_samples_split"}
        return GradientBoostingClassifier(
            **{k: v for k, v in tree_params.items() if k in allowed},
            random_state=42,
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    else:
        return RandomForestClassifier(**tree_params, random_state=42)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: class distribution check
    class_dist = y_train.value_counts(normalize=True).sort_index()
    print("Class distribution:")
    for cls, ratio in class_dist.items():
        flag = " <-- WARNING: below 10% threshold" if ratio < 0.10 else ""
        print(f"  Class {cls}: {ratio:.1%}{flag}")

    model_type = params.get("model_type", "random_forest")

    with mlflow.start_run():
        mlflow.log_params(params)

        model = _build_model(params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)

        # Bonus 5: include class distribution in metrics.json
        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc,
                "f1_score": f1,
                "class_distribution": {str(k): round(v, 4) for k, v in class_dist.items()},
            }, f, indent=2)

        # Bonus 3: per-class precision/recall + confusion matrix
        cm = confusion_matrix(y_eval, preds)
        report = classification_report(y_eval, preds, target_names=["low", "medium", "high"])
        with open("outputs/report.txt", "w") as f:
            f.write(f"Model: {model_type}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\nClassification Report (precision / recall per class):\n")
            f.write(report)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
