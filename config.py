"""Shared constants: feature lists, category codes and file paths."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_CSV = os.path.join(BASE_DIR, 'heart.csv')
CLEAN_CSV = os.path.join(BASE_DIR, 'heart_cleaned.csv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'metrics.json')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

TARGET = 'target'

# Continuous / count features -> median imputation + StandardScaler
NUMERIC = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Nominal features -> most-frequent imputation + one-hot encoding
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

FEATURES = NUMERIC + CATEGORICAL

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Minimum recall we want when choosing the decision threshold.
# Missing a sick patient (false negative) costs far more than a false alarm.
TARGET_RECALL = 0.90

# Category codes exactly as they appear in the UCI Cleveland file
CATEGORY_LABELS = {
    'sex': {0: 'Female', 1: 'Male'},
    'cp': {1: 'Typical angina', 2: 'Atypical angina', 3: 'Non-anginal pain', 4: 'Asymptomatic'},
    'fbs': {0: 'Fasting blood sugar <= 120 mg/dl', 1: 'Fasting blood sugar > 120 mg/dl'},
    'restecg': {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'},
    'exang': {0: 'No', 1: 'Yes'},
    'slope': {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'},
    'thal': {3: 'Normal', 6: 'Fixed defect', 7: 'Reversible defect'},
}

# Plausible physiological ranges used for validation in clean_data.py and the web form
NUMERIC_RANGES = {
    'age': (18, 100),
    'trestbps': (80, 220),
    'chol': (100, 600),
    'thalach': (60, 220),
    'oldpeak': (0.0, 7.0),
    'ca': (0, 3),
}

FEATURE_DESCRIPTIONS = {
    'age': 'Age (years)',
    'sex': 'Sex',
    'cp': 'Chest pain type',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar',
    'restecg': 'Resting ECG result',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise-induced angina',
    'oldpeak': 'ST depression induced by exercise',
    'slope': 'Slope of peak exercise ST segment',
    'ca': 'Major vessels coloured by fluoroscopy (0-3)',
    'thal': 'Thallium stress test result',
}
