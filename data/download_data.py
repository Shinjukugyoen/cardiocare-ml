"""
download_data.py
Download the UCI Heart Disease (Cleveland) dataset, binarize the target,
and save a clean CSV to data/heart_disease.csv.

Source: UCI ML Repository, Heart Disease (id=45), Cleveland processed
        subset = 303 rows, 13 features + 1 target.
"""

from pathlib import Path
import pandas as pd

# Save the CSV right next to this script (i.e. inside data/).
OUT_PATH = Path(__file__).resolve().parent / "heart_disease.csv"

# Try the official UCI file first; fall back to a verified mirror.
URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://raw.githubusercontent.com/ywchiu/cdc_course/master/data/processed.cleveland.data.csv",
]

# The raw file has no header row, so we name the 14 columns ourselves.
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def load_raw() -> pd.DataFrame:
    """Try each source URL in order until one works. '?' = missing value."""
    last_error = None
    for url in URLS:
        try:
            df = pd.read_csv(url, header=None, names=COLUMNS, na_values="?")
            print(f"[ok] downloaded {len(df)} rows from: {url}")
            return df
        except Exception as e:
            print(f"[warn] source failed: {url}")
            last_error = e
    raise RuntimeError(f"All download sources failed. Last error: {last_error}")


def main() -> None:
    df = load_raw()

    # Binarize the target: num in {0,1,2,3,4} -> 0 (no disease) / 1 (disease).
    df["target"] = (df["num"] > 0).astype(int)

    # Drop the original multiclass target to prevent label leakage.
    df = df.drop(columns=["num"])

    # Write a clean CSV without the pandas row index.
    df.to_csv(OUT_PATH, index=False)

    print(f"[ok] saved -> {OUT_PATH}")
    print(f"     shape: {df.shape}")
    print("     target distribution (normalized):")
    print(df["target"].value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()