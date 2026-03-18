import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
            'exang', 'oldpeak', 'slope', 'ca', 'thal']
TARGET = 'target'

def clean_heart_data(input_csv='heart.csv', output_csv='heart_cleaned.csv'):
    """Clean heart disease dataset:
    - Handle missing values (drop rows with NaN, fill common cases)
    - Convert categorical columns (already numeric in standard dataset)
    - Normalize numerical features using StandardScaler
    """
    if not os.path.exists(input_csv):
        print(f'{input_csv} not found!')
        return
    
    print('Loading data...')
    df = pd.read_csv(input_csv)
    
    # Verify columns
    expected_cols = FEATURES + [TARGET]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f'Missing columns: {missing_cols}')
        return
    
    df = df[expected_cols]
    
    print('Original shape:', df.shape)
    
    # 1. Handle missing values
    print('Handling missing values...')
    nan_count = df.isnull().sum()
    if nan_count.any():
        print('NaNs per column:\n', nan_count[nan_count > 0])
        # For ca and thal (common missing in UCI), fill mode
        for col in ['ca', 'thal']:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        # Drop remaining rows with NaN
        df = df.dropna()
    else:
        print('No missing values found.')
    
    print('After missing values handling:', df.shape)
    
    # 2. Categorical columns (ensure int)
    categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in categorical:
        df[col] = df[col].astype(int)
    
    # 3. Normalize numerical columns
    print('Normalizing numerical features...')
    numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    df[numerical] = scaler.fit_transform(df[numerical])
    
    # Save scaler for later use
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save cleaned data
    df.to_csv(output_csv, index=False)
    print(f'Cleaned data saved to {output_csv}')
    print('Cleaning complete!')
    
    # Summary stats
    print('\nSummary after cleaning:')
    print(df.describe())

if __name__ == '__main__':
    clean_heart_data()

