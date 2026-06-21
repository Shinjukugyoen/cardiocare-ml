"""
test_pipeline.py
Four unit tests for the heart-disease ML pipeline.

Run from the repo root:  python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import clean_data, split_features_target, build_preprocessor
from inference import validate_feature_ranges
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def _load_xy():
    df = clean_data(pd.read_csv(ROOT / "data" / "heart_disease.csv"))
    return split_features_target(df)


def _build_fitted_pipeline(X_train, y_train) -> Pipeline:
    """A small, deterministic pipeline used across the tests."""
    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = _load_xy()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        cls.pipe = _build_fitted_pipeline(cls.X_train, cls.y_train)

    def test_prediction_shape_matches_input(self):
        """The number of predictions must equal the number of input rows."""
        preds = self.pipe.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

    def test_probabilities_in_range_and_sum_to_one(self):
        """Each predicted probability is in [0, 1] and each row sums to ~1."""
        proba = self.pipe.predict_proba(self.X_test)
        self.assertTrue(((proba >= 0.0) & (proba <= 1.0)).all())
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_input_range_validation(self):
        """Valid input passes; an out-of-range chol value is flagged."""
        self.assertEqual(validate_feature_ranges(self.X_test), [])
        bad = self.X_test.copy()
        bad.iloc[0, bad.columns.get_loc("chol")] = 9999   # impossible cholesterol
        self.assertNotEqual(validate_feature_ranges(bad), [])

    def test_pipeline_is_deterministic(self):
        """Same data + same seed must give identical predictions."""
        p1 = _build_fitted_pipeline(self.X_train, self.y_train).predict(self.X_test)
        p2 = _build_fitted_pipeline(self.X_train, self.y_train).predict(self.X_test)
        np.testing.assert_array_equal(p1, p2)


if __name__ == "__main__":
    unittest.main()