import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Expected columns
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
            'exang', 'oldpeak', 'slope', 'ca', 'thal']
TARGET = 'target'

def train_and_save_model(csv_path='heart_cleaned.csv'):
    """Train model on CLEANED/SCALED data + scaler for consistency."""
    if not os.path.exists(csv_path):
        print(f'{csv_path} not found. Run clean_data.py first.')
        return
    
    df = pd.read_csv(csv_path)
    expected_cols = FEATURES + [TARGET]
    if not all(col in df.columns for col in expected_cols):
        print(f'Missing columns. Expected: {expected_cols}')
        return
    
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.3f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, 'heart_model.pkl')
    print('Model saved as heart_model.pkl')
    print('Feature Importances:')
    importances = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)

if __name__ == '__main__':
    train_and_save_model()

