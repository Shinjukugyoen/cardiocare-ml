"""
train.py  (full: 3 models + MLflow + 5-fold CV + hyperparameter search + save)
Run from the repo root:  python src/train.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import pandas as pd, joblib
import mlflow, mlflow.sklearn
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import clean_data, split_features_target, build_preprocessor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def find_data_file(filename: str = "heart_disease.csv") -> Path:
    for parent in Path(__file__).resolve().parents:
        c = parent / "data" / filename
        if c.exists():
            return c
    raise FileNotFoundError(filename)


def make_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("select", SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))),
        ("clf", classifier),
    ])


def evaluate(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def main() -> None:
    df = clean_data(pd.read_csv(find_data_file()))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"train={X_train.shape}  test={X_test.shape}\n")

    models = {
        "logreg": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), "linear"),
        "svc": (SVC(probability=True, random_state=RANDOM_STATE), "kernel"),
        "random_forest": (RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE), "tree"),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mlflow.set_experiment("cardiocare")
    cv_scores = {}

    print(f"{'model':14s} {'cv_bal':>7s} {'test_bal':>8s} {'recall':>7s}  conf[tn,fp,fn,tp]")
    for name, (clf, family) in models.items():
        pipe = make_pipeline(clf)
        cv_bal = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="balanced_accuracy").mean()
        pipe.fit(X_train, y_train)
        m = evaluate(pipe, X_test, y_test)
        cv_scores[name] = cv_bal
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_family", family)
            mlflow.log_params(clf.get_params())
            mlflow.log_metric("cv_balanced_accuracy", cv_bal)
            for k, v in m.items():
                mlflow.log_metric(k, float(v))
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        print(f"{name:14s} {cv_bal:7.4f} {m['balanced_accuracy']:8.4f} {m['recall']:7.3f}  [{m['tn']},{m['fp']},{m['fn']},{m['tp']}]")

    mask = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)).fit(X_train, y_train).named_steps["select"].get_support()
    print(f"\nselected features ({mask.sum()}/{len(mask)}):", list(X.columns[mask]))

    best_name = max(cv_scores, key=cv_scores.get)
    grids = {"logreg": {"clf__C": [0.01, 0.1, 1, 10]},
             "svc": {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]},
             "random_forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 5, 10]}}
    base = {"logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "svc": SVC(probability=True, random_state=RANDOM_STATE),
            "random_forest": RandomForestClassifier(random_state=RANDOM_STATE)}[best_name]

    print(f"\ntuning best CV model: {best_name}")
    gs = GridSearchCV(make_pipeline(base), grids[best_name], cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned = gs.best_estimator_
    tm = evaluate(tuned, X_test, y_test)
    with mlflow.start_run(run_name=f"{best_name}_tuned"):
        mlflow.set_tag("model_family", models[best_name][1])
        mlflow.set_tag("tuned", "true")
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("cv_balanced_accuracy", gs.best_score_)
        for k, v in tm.items():
            mlflow.log_metric(k, float(v))
        mlflow.sklearn.log_model(tuned, name="model", serialization_format="cloudpickle")
    print(f"best params: {gs.best_params_}  cv_bal={gs.best_score_:.4f}")
    print(f"tuned test:  bal={tm['balanced_accuracy']:.4f}  recall={tm['recall']:.3f}  fn={tm['fn']}")

    models_dir = find_data_file().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(tuned, models_dir / "final_model.joblib")
    print(f"saved -> {models_dir / 'final_model.joblib'}")


if __name__ == "__main__":
    main()