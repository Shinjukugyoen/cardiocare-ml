"""
inference.py
Load the trained model and run predictions on a batch CSV of patients.

Usage:  python src/inference.py data/sample_input.csv

Logs every inference call (timestamp, model version, input shape, predictions,
and accuracy when the true label is present) to logs/inference.log.
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "final_model.joblib"
MODEL_VERSION = "final_model_v1"

FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Clinically plausible ranges for the continuous features.
FEATURE_BOUNDS = {
    "age": (0, 120), "trestbps": (0, 300), "chol": (0, 600),
    "thalach": (0, 250), "oldpeak": (0, 10),
}

# Log inference calls to a file (created on first run).
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "inference.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def load_model(path: Path = MODEL_PATH):
    """Load the saved sklearn pipeline from disk."""
    return joblib.load(path)


def validate_feature_ranges(X: pd.DataFrame) -> list[str]:
    """Return a list of range problems; an empty list means the input is valid."""
    problems = []
    for col, (low, high) in FEATURE_BOUNDS.items():
        if col in X.columns:
            n_bad = int(((X[col] < low) | (X[col] > high)).sum())
            if n_bad:
                problems.append(f"{col} outside [{low}, {high}] in {n_bad} row(s)")
    return problems


def run_inference(input_csv: Path) -> pd.DataFrame:
    """Predict disease for each patient row in input_csv."""
    model = load_model()
    raw = pd.read_csv(input_csv)
    X = raw[[c for c in FEATURE_COLUMNS if c in raw.columns]]

    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]   # probability of class 1 (disease)

    # Inference logging (timestamp is added by the logging format).
    msg = f"model={MODEL_VERSION} input_shape={X.shape} predictions={[int(p) for p in preds]}"
    if "target" in raw.columns:
        acc = float((preds == raw["target"]).mean())
        msg += f" accuracy={acc:.4f}"
    logging.info(msg)

    out = X.copy()
    out["prediction"] = preds
    out["disease_probability"] = proba.round(4)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <input_csv>")
        sys.exit(1)
    result = run_inference(Path(sys.argv[1]))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()