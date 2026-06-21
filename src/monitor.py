"""
monitor.py
Data-drift monitoring for the heart-disease model.

Pipeline:
  1. Inference logging is already wired into inference.py; this script adds
     drift-specific logging to logs/monitor.log.
  2. Build an artificial 'drifted' copy of the test set that simulates a
     population / measurement shift: cholesterol and resting BP rise, max heart
     rate falls, ST-depression rises (age is left UNCHANGED as a control).
  3. Run a Kolmogorov-Smirnov (KS) test per continuous feature, train vs drifted.
     Flag any feature with p-value < 0.05 as drifted.
  4. Compare balanced accuracy on the original vs drifted test set, linking
     input drift to performance degradation.
  5. Plot a metric-over-time line chart using synthetic timestamps.

Run from the repo root:  python src/monitor.py
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")            # headless backend (no display window needed)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import clean_data, split_features_target

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "final_model.joblib"
RANDOM_STATE = 42
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Each drifted feature: (mean shift, added noise std). 'age' is intentionally
# left out so it acts as a control that should NOT be flagged.
DRIFT_SHIFTS = {
    "chol": (50, 15),       # cholesterol readings rise
    "trestbps": (25, 10),   # resting blood pressure rises
    "oldpeak": (1.5, 0.5),  # ST depression rises
    "thalach": (-30, 10),   # max heart rate falls
}

logging.basicConfig(
    filename=ROOT / "logs" / "monitor.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def make_drift(X: pd.DataFrame) -> pd.DataFrame:
    """Return a drifted copy of X by shifting several continuous features."""
    np.random.seed(RANDOM_STATE)
    Xd = X.copy()
    for col, (shift, noise) in DRIFT_SHIFTS.items():
        Xd[col] = Xd[col] + shift + np.random.normal(0, noise, size=len(Xd))
    return Xd


def main() -> None:
    df = clean_data(pd.read_csv(ROOT / "data" / "heart_disease.csv"))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    X_test_drift = make_drift(X_test)

    # --- KS test per continuous feature: train (reference) vs drifted test ---
    print(f"{'feature':10s} {'ks_stat':>8s} {'p_value':>9s}  drift?")
    for col in CONTINUOUS:
        stat, p = ks_2samp(X_train[col], X_test_drift[col])
        flagged = bool(p < 0.05)
        logging.info(f"KS feature={col} stat={stat:.4f} p={p:.4f} drift={flagged}")
        print(f"{col:10s} {stat:8.4f} {p:9.4f}  {'YES' if flagged else 'no'}")

    # --- performance: original test vs drifted test ---
    model = joblib.load(MODEL_PATH)
    bal_orig = balanced_accuracy_score(y_test, model.predict(X_test))
    bal_drift = balanced_accuracy_score(y_test, model.predict(X_test_drift))
    print(f"\nbalanced_accuracy  original={bal_orig:.4f}  drifted={bal_drift:.4f}"
          f"  drop={bal_orig - bal_drift:.4f}")
    logging.info(f"perf original={bal_orig:.4f} drifted={bal_drift:.4f} "
                 f"drop={bal_orig - bal_drift:.4f}")

    # --- metric-over-time line chart (synthetic timestamps) ---
    days = pd.date_range("2026-06-01", periods=10, freq="D")
    accs = np.linspace(bal_orig, bal_drift, len(days))   # simulated gradual decay
    plt.figure(figsize=(8, 4))
    plt.plot(days, accs, marker="o", label="balanced accuracy")
    plt.axhline(bal_orig, color="green", linestyle="--", label="baseline")
    plt.axhline(0.5, color="red", linestyle=":", label="random guess")
    plt.title("Balanced accuracy over time (simulated drift)")
    plt.ylabel("balanced accuracy"); plt.xlabel("date")
    plt.legend(); plt.tight_layout()
    out_path = ROOT / "logs" / "drift_metric_over_time.png"
    plt.savefig(out_path)
    print(f"saved chart -> {out_path}")


if __name__ == "__main__":
    main()