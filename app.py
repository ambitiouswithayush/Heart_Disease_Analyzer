"""Flask app: interactive dashboard, risk predictor and model metrics."""
import json
import os

import joblib
import pandas as pd
import plotly.express as px
from flask import Flask, jsonify, render_template, request

from config import (CATEGORICAL, CATEGORY_LABELS, CLEAN_CSV, FEATURE_DESCRIPTIONS, FEATURES, METRICS_PATH,
                    MODEL_PATH, NUMERIC_RANGES, TARGET)

app = Flask(__name__)

_cache = {}


def load_artifacts():
    """Load the trained pipeline and metrics once (lazy, so tests can run without them)."""
    if 'model' not in _cache:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError('Model not found. Run `python run_pipeline.py` first.')
        _cache['model'] = joblib.load(MODEL_PATH)
        with open(METRICS_PATH) as f:
            _cache['metrics'] = json.load(f)
    return _cache['model'], _cache['metrics']


def load_data():
    df = pd.read_csv(CLEAN_CSV)
    df['Status'] = df[TARGET].map({0: 'No disease', 1: 'Disease'})
    return df


@app.route('/')
@app.route('/dashboard')
def dashboard():
    return render_template('index.html', has_data=os.path.exists(CLEAN_CSV), labels=CATEGORY_LABELS)


@app.route('/api/data')
def api_data():
    """Charts for the dashboard, filtered server-side by age range, sex and chest-pain type."""
    if not os.path.exists(CLEAN_CSV):
        return jsonify({'error': 'Run the pipeline first'}), 404

    df = load_data()
    age_min = request.args.get('age_min', type=int)
    age_max = request.args.get('age_max', type=int)
    sex = request.args.get('sex', type=int)
    cp = request.args.get('cp', type=int)
    if age_min is not None:
        df = df[df['age'] >= age_min]
    if age_max is not None:
        df = df[df['age'] <= age_max]
    if sex is not None:
        df = df[df['sex'] == sex]
    if cp is not None:
        df = df[df['cp'] == cp]

    if df.empty:
        return jsonify({'count': 0, 'disease_rate': None, 'charts': {}})

    colors = {'No disease': '#4C9BE8', 'Disease': '#E8604C'}
    named = df.assign(Sex=df['sex'].map(CATEGORY_LABELS['sex']),
                      ChestPain=df['cp'].map(CATEGORY_LABELS['cp']))
    rate_by_cp = named.groupby('ChestPain')[TARGET].mean().mul(100).round(1).reset_index()
    corr = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', TARGET]].corr().round(2)

    charts = {
        'status': px.pie(named, names='Status', color='Status', color_discrete_map=colors,
                         title='Disease distribution'),
        'age': px.violin(named, x='Status', y='age', color='Status', box=True, color_discrete_map=colors,
                         title='Age by disease status'),
        'thalach': px.box(named, x='Status', y='thalach', color='Status', color_discrete_map=colors,
                          title='Max heart rate by disease status'),
        'sex': px.histogram(named, x='Sex', color='Status', barmode='group', color_discrete_map=colors,
                            title='Disease by sex'),
        'cp': px.bar(rate_by_cp, x='ChestPain', y=TARGET, title='Disease rate (%) by chest pain type',
                     labels={TARGET: 'Disease rate (%)', 'ChestPain': ''}),
        'corr': px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                          title='Correlation matrix'),
    }
    return jsonify({
        'count': int(len(df)),
        'disease_rate': round(float(df[TARGET].mean()) * 100, 1),
        'charts': {k: json.loads(v.to_json()) for k, v in charts.items()},
    })


def parse_patient(form):
    """Validate form input and return a one-row DataFrame in training column order."""
    row, errors = {}, []
    for f in FEATURES:
        raw = form.get(f, '').strip()
        if raw == '':
            errors.append(f'{FEATURE_DESCRIPTIONS[f]} is required')
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f'{FEATURE_DESCRIPTIONS[f]} must be a number')
            continue
        if f in CATEGORICAL and int(value) not in CATEGORY_LABELS[f]:
            errors.append(f'{FEATURE_DESCRIPTIONS[f]}: unknown code {raw}')
        elif f in NUMERIC_RANGES and not NUMERIC_RANGES[f][0] <= value <= NUMERIC_RANGES[f][1]:
            lo, hi = NUMERIC_RANGES[f]
            errors.append(f'{FEATURE_DESCRIPTIONS[f]} must be between {lo} and {hi}')
        row[f] = value
    return pd.DataFrame([row], columns=FEATURES), errors


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    ctx = {'labels': CATEGORY_LABELS, 'ranges': NUMERIC_RANGES, 'descriptions': FEATURE_DESCRIPTIONS,
           'values': request.form if request.method == 'POST' else {}}
    try:
        artifacts, metrics = load_artifacts()
    except FileNotFoundError as e:
        return render_template('predict.html', error=str(e), **ctx)
    ctx['model_name'] = artifacts['model_name']
    ctx['threshold'] = artifacts['threshold']

    if request.method == 'POST':
        patient, errors = parse_patient(request.form)
        if errors:
            return render_template('predict.html', error='; '.join(errors), **ctx)
        probability = float(artifacts['pipeline'].predict_proba(patient)[0, 1])
        ctx['probability'] = probability
        ctx['high_risk'] = probability >= artifacts['threshold']
    return render_template('predict.html', **ctx)


@app.route('/metrics')
def metrics_page():
    try:
        _, metrics = load_artifacts()
    except FileNotFoundError as e:
        return str(e), 404
    return render_template('metrics.html', m=metrics, descriptions=FEATURE_DESCRIPTIONS)


@app.route('/api/metrics')
def api_metrics():
    _, metrics = load_artifacts()
    return jsonify(metrics)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)),
            debug=os.environ.get('FLASK_DEBUG') == '1')
