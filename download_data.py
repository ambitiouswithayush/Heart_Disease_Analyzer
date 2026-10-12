#!/usr/bin/env python3
"""Download the UCI Cleveland Heart Disease dataset (303 records)."""
import os
import requests

from config import RAW_CSV

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
HEADER = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target'


def download_heart_dataset(force=False):
    """Save the raw file as CSV.

    - Missing values ('?') are kept as empty cells so they can be imputed
      later *inside* the model pipeline (dropping them would lose 6 patients).
    - The original target is 0-4 (disease severity); it is binarised to
      0 = no disease, 1 = disease (any severity).
    """
    if os.path.exists(RAW_CSV) and not force:
        print(f'{RAW_CSV} already exists. Use force=True to re-download.')
        return

    print('Downloading UCI Cleveland heart disease data...')
    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    rows = []
    for line in response.text.strip().splitlines():
        values = ['' if v.strip() == '?' else v.strip() for v in line.split(',')]
        values[-1] = '1' if int(float(values[-1])) > 0 else '0'
        rows.append(','.join(values))

    with open(RAW_CSV, 'w') as f:
        f.write(HEADER + '\n' + '\n'.join(rows) + '\n')

    print(f'Saved {len(rows)} records to {RAW_CSV}')


if __name__ == '__main__':
    download_heart_dataset(force=True)
