"""Train, compare and evaluate models without data leakage.

Workflow
1. Stratified 80/20 split. The test set is locked away until the very end.
2. All preprocessing (imputation, scaling, one-hot) sits inside a sklearn
   Pipeline, so in every CV fold it is fitted on that fold's training part only.
3. Candidate models are tuned with GridSearchCV (5-fold stratified CV, ROC-AUC)
   on the training set and compared against a majority-class baseline.
4. The decision threshold is chosen from out-of-fold training predictions so
   that recall >= TARGET_RECALL (missing a sick patient is the costly error).
5. The chosen model is evaluated ONCE on the untouched test set.
"""
import json
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (ARTIFACTS_DIR, CATEGORICAL, CLEAN_CSV, CV_FOLDS, FEATURES, IMAGES_DIR, METRICS_PATH,
                    MODEL_PATH, NUMERIC, RANDOM_STATE, TARGET, TARGET_RECALL, TEST_SIZE)


def build_preprocessor():
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer([
        ('num', numeric, NUMERIC),
        ('cat', categorical, CATEGORICAL),
    ])


def candidate_models():
    """(name, estimator, hyper-parameter grid) for each model we compare."""
    return [
        ('Baseline (majority class)', DummyClassifier(strategy='most_frequent'), {}),
        ('Logistic Regression', LogisticRegression(max_iter=2000),
         {'model__C': [0.01, 0.1, 1, 10]}),
        ('Random Forest', RandomForestClassifier(random_state=RANDOM_STATE),
         {'model__n_estimators': [200], 'model__max_depth': [3, 5, None],
          'model__min_samples_leaf': [1, 5]}),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=RANDOM_STATE),
         {'model__n_estimators': [100], 'model__learning_rate': [0.05, 0.1],
          'model__max_depth': [2, 3]}),
    ]


def make_pipeline(estimator):
    return Pipeline([('preprocess', build_preprocessor()), ('model', estimator)])


def pick_threshold(y_true, proba, target_recall=TARGET_RECALL):
    """Highest threshold whose recall is still >= target_recall.

    A higher threshold means fewer false alarms, so among thresholds that
    meet the recall target we take the largest one.
    """
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = recall[:-1] >= target_recall
    return float(thresholds[ok].max()) if ok.any() else 0.5


def scores(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        'threshold': round(float(threshold), 3),
        'accuracy': round(float(accuracy_score(y_true, pred)), 3),
        'precision': round(float(precision_score(y_true, pred, zero_division=0)), 3),
        'recall': round(float(recall_score(y_true, pred)), 3),
        'specificity': round(float(tn / (tn + fp)), 3),
        'f1': round(float(f1_score(y_true, pred)), 3),
        'roc_auc': round(float(roc_auc_score(y_true, proba)), 3),
        'brier': round(float(brier_score_loss(y_true, proba)), 3),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


def save_plots(pipeline, X_test, y_test, proba, threshold, importances):
    os.makedirs(IMAGES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name='Test set')
    ax.plot([0, 1], [0, 1], '--', color='grey', label='Random guess')
    ax.legend()
    ax.set_title('ROC curve (held-out test set)')
    fig.savefig(os.path.join(IMAGES_DIR, '09_roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, t, title in [(axes[0], 0.5, 'Default threshold 0.5'),
                         (axes[1], threshold, f'Tuned threshold {threshold:.2f}')]:
        ConfusionMatrixDisplay.from_predictions(y_test, (proba >= t).astype(int), ax=ax, colorbar=False,
                                                display_labels=['No disease', 'Disease'], cmap='Blues')
        ax.set_title(title)
    fig.savefig(os.path.join(IMAGES_DIR, '10_confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    imp = pd.Series(importances).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color='steelblue')
    ax.set_xlabel('Drop in test ROC-AUC when the feature is shuffled')
    ax.set_title('Permutation feature importance')
    fig.savefig(os.path.join(IMAGES_DIR, '11_permutation_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=5, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, 'o-', label='Model')
    ax.plot([0, 1], [0, 1], '--', color='grey', label='Perfectly calibrated')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed disease rate')
    ax.set_title('Calibration curve (test set)')
    ax.legend()
    fig.savefig(os.path.join(IMAGES_DIR, '12_calibration.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_and_save_model(csv_path=CLEAN_CSV):
    df = pd.read_csv(csv_path)
    X, y = df[FEATURES], df[TARGET]

    # 1. Lock away the test set (stratified keeps the disease ratio equal in both parts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f'Train: {len(X_train)} rows | Test: {len(X_test)} rows | Disease rate: {y.mean():.1%}')

    # 2-3. Tune and compare models with cross-validation on the training set only
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    comparison, searches = [], {}
    for name, estimator, grid in candidate_models():
        search = GridSearchCV(make_pipeline(estimator), grid, cv=cv,
                              scoring=['roc_auc', 'accuracy', 'recall', 'f1'], refit='roc_auc')
        search.fit(X_train, y_train)
        i = search.best_index_
        res = search.cv_results_
        comparison.append({
            'model': name,
            'cv_roc_auc': round(float(res['mean_test_roc_auc'][i]), 3),
            'cv_roc_auc_std': round(float(res['std_test_roc_auc'][i]), 3),
            'cv_accuracy': round(float(res['mean_test_accuracy'][i]), 3),
            'cv_recall': round(float(res['mean_test_recall'][i]), 3),
            'cv_f1': round(float(res['mean_test_f1'][i]), 3),
            'best_params': {k.replace('model__', ''): v for k, v in search.best_params_.items()},
        })
        searches[name] = search

    comparison_df = pd.DataFrame(comparison).set_index('model')
    print('\nCross-validation results (training set, 5 folds):')
    print(comparison_df.drop(columns='best_params').to_string())

    # Pick the best model by mean CV ROC-AUC
    best_name = comparison_df['cv_roc_auc'].idxmax()
    best_pipeline = searches[best_name].best_estimator_
    print(f'\nSelected model: {best_name} {comparison_df.loc[best_name, "best_params"]}')

    # 4. Choose the decision threshold from out-of-fold predictions (never from the test set)
    oof_proba = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    threshold = pick_threshold(y_train, oof_proba)
    print(f'Decision threshold for recall >= {TARGET_RECALL:.0%}: {threshold:.3f}')

    # 5. Final, one-time evaluation on the untouched test set
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_default = scores(y_test, test_proba, 0.5)
    test_tuned = scores(y_test, test_proba, threshold)
    print('\nTest set @0.5:  ', {k: v for k, v in test_default.items() if k != 'confusion_matrix'})
    print('Test set @tuned:', {k: v for k, v in test_tuned.items() if k != 'confusion_matrix'})
    print('Confusion matrix @tuned:', test_tuned['confusion_matrix'])

    # Model-agnostic importance on the raw (pre-encoding) features
    perm = permutation_importance(best_pipeline, X_test, y_test, scoring='roc_auc',
                                  n_repeats=30, random_state=RANDOM_STATE)
    importances = {f: round(float(v), 4) + 0.0 for f, v in zip(FEATURES, perm.importances_mean)}
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
    print('\nPermutation importance (drop in test ROC-AUC):', importances)

    # Persist the WHOLE pipeline (preprocessing + model) so serving == training
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump({'pipeline': best_pipeline, 'threshold': threshold, 'model_name': best_name}, MODEL_PATH)

    metrics = {
        'dataset': {'rows': int(len(df)), 'features': len(FEATURES), 'disease_rate': round(float(y.mean()), 3),
                    'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)),
                    'missing_values': {k: int(v) for k, v in df[FEATURES].isnull().sum().items() if v}},
        'cv_comparison': comparison,
        'selected_model': best_name,
        'target_recall': TARGET_RECALL,
        'test_default_threshold': test_default,
        'test_tuned_threshold': test_tuned,
        'permutation_importance': importances,
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    save_plots(best_pipeline, X_test, y_test, test_proba, threshold, importances)
    print(f'\nSaved {MODEL_PATH}, {METRICS_PATH} and evaluation plots in {IMAGES_DIR}/')
    return metrics


if __name__ == '__main__':
    train_and_save_model()
