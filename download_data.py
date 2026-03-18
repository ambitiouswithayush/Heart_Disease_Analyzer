#!/usr/bin/env python3
import requests
import os

def download_heart_dataset():
    """Download UCI Heart Disease dataset (heart.csv) - 303 records."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    
    filename = 'heart.csv'
    if os.path.exists(filename):
        print(f'{filename} already exists. Size: {os.path.getsize(filename)} bytes')
        return
    
    print('Downloading heart.csv from UCI repository...')
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse UCI format (no header, 14 cols incl target as 0-4 -> 0/1 binary)
        lines = response.text.strip().split('\n')
        data = []
        for line in lines:
            if '?' not in line:  # Skip missing values for simplicity
                row = line.split(',')
                row[-1] = '1' if int(float(row[-1])) > 0 else '0'  # Binary target
                data.append(','.join(row))
        
        with open(filename, 'w') as f:
            # Header
            f.write('age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n')
            f.write('\n'.join(data))
        
        print(f'✅ Downloaded {len(data)} clean records to {filename}')
        
    except Exception as e:
        print(f'❌ Download failed: {e}')
        print('Alternative: Manual download from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset')

if __name__ == '__main__':
    download_heart_dataset()

