"""Tests for the leakage fix, the saved model and the web app.

Run with: python -m pytest -q   (after python run_pipeline.py)
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLEAN_CSV, FEATURES, MODEL_PATH, NUMERIC, RANDOM_STATE, TARGET  # noqa: E402
from train_model import make_pipeline, pick_threshold  # noqa: E402

needs_pipeline = pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason='run python run_pipeline.py first')


@pytest.fixture(scope='module')
def data():
    df = pd.read_csv(CLEAN_CSV)
    return train_test_split(df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET],
                            random_state=RANDOM_STATE)


def test_cleaned_data_is_not_scaled():
    """Scaling must happen inside the pipeline, not in the cleaned CSV (the original leakage bug)."""
    df = pd.read_csv(CLEAN_CSV)
    assert df['age'].between(18, 100).all()
    assert df['chol'].mean() > 100


def test_scaler_is_fitted_on_training_rows_only(data):
    X_train, X_test, y_train, _ = data
    pipe = make_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    scaler = pipe.named_steps['preprocess'].named_transformers_['num'].named_steps['scale']
    train_means = X_train[NUMERIC].fillna(X_train[NUMERIC].median()).mean().values
    full_means = pd.concat([X_train, X_test])[NUMERIC].fillna(X_train[NUMERIC].median()).mean().values
    assert np.allclose(scaler.mean_, train_means)
    assert not np.allclose(scaler.mean_, full_means)


def test_pipeline_handles_missing_values(data):
    X_train, X_test, y_train, _ = data
    pipe = make_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    row = X_test.iloc[[0]].copy()
    row[['ca', 'thal']] = np.nan
    proba = pipe.predict_proba(row)[0, 1]
    assert 0 <= proba <= 1


def test_pick_threshold_meets_recall():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    t = pick_threshold(y, p, target_recall=0.85)
    recall = ((p >= t) & (y == 1)).sum() / y.sum()
    assert recall >= 0.85


@needs_pipeline
def test_saved_model_beats_baseline(data):
    _, X_test, _, y_test = data
    artifacts = joblib.load(MODEL_PATH)
    proba = artifacts['pipeline'].predict_proba(X_test)[:, 1]
    accuracy = ((proba >= 0.5) == y_test).mean()
    baseline = max(y_test.mean(), 1 - y_test.mean())
    assert accuracy > baseline + 0.15
    assert 0 < artifacts['threshold'] < 1


@needs_pipeline
class TestApp:
    @pytest.fixture
    def client(self):
        from app import app
        app.testing = True
        return app.test_client()

    def form(self, **overrides):
        values = {'age': 60, 'sex': 1, 'cp': 4, 'trestbps': 140, 'chol': 290, 'fbs': 0, 'restecg': 2,
                  'thalach': 110, 'exang': 1, 'oldpeak': 2.5, 'slope': 2, 'ca': 2, 'thal': 7}
        values.update(overrides)
        return {k: str(v) for k, v in values.items()}

    def test_pages_load(self, client):
        for url in ['/', '/predict', '/metrics', '/api/metrics']:
            assert client.get(url).status_code == 200

    def test_dashboard_filters(self, client):
        all_rows = client.get('/api/data').get_json()
        women = client.get('/api/data?sex=0').get_json()
        assert all_rows['count'] == 303
        assert 0 < women['count'] < all_rows['count']
        assert women['disease_rate'] < all_rows['disease_rate']

    def test_high_risk_patient(self, client):
        html = client.post('/predict', data=self.form()).get_data(as_text=True)
        assert 'HIGH RISK' in html

    def test_low_risk_patient(self, client):
        low = self.form(age=40, sex=0, cp=3, chol=200, thalach=175, exang=0, oldpeak=0, slope=1, ca=0, thal=3,
                        restecg=0, trestbps=120)
        html = client.post('/predict', data=low).get_data(as_text=True)
        assert 'Low risk' in html

    def test_invalid_input_rejected(self, client):
        html = client.post('/predict', data=self.form(cp=9, age=500)).get_data(as_text=True)
        assert 'unknown code' in html and 'must be between' in html
