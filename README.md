# CardioCare ML — Heart Disease Prediction

End-to-end ML system predicting heart disease from the UCI Heart Disease
(Cleveland) dataset. Supports a clinician's decision — it informs, not decides.

## Environment
- Python 3.10.x
- See `requirements.txt` (pinned versions)

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
```

## Reproduce the full pipeline
```bash
# 1. Download data -> data/heart_disease.csv
python data/download_data.py

# 2. Train 3 models, log to MLflow, tune, save final model
python src/train.py

# 3. View experiments
mlflow ui            # http://localhost:5000

# 4. Run unit tests
python -m unittest discover -s tests -v

# 5. Build & run the inference container
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0

# 6. Drift monitoring
python src/monitor.py
```

## Structure
- `data/` — download script + dataset
- `notebooks/01_eda_preprocessing.ipynb` — EDA & preprocessing
- `src/preprocessing.py` — leakage-safe preprocessing pipeline
- `src/train.py` — training, MLflow, CV, tuning
- `src/inference.py` — inference entry point (Docker)
- `src/monitor.py` — KS drift detection
- `tests/test_pipeline.py` — 4 unit tests
- `Dockerfile`, `.github/workflows/ci.yml`

## Final model
Tuned Logistic Regression (C=0.1), chosen for high recall (minimizing false
negatives) in a clinical context. Saved to `models/final_model.joblib`.