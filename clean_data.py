"""Validate and tidy the raw data.

This step deliberately does NOT impute or scale. Imputation and scaling learn
statistics (median, mean, std) from the data, so they must be fitted on the
training split only. They live inside the sklearn Pipeline in train_model.py.
Doing them here, on the full dataset, would leak test-set information into
training (the bug in the first version of this project).
"""
import pandas as pd

from config import CATEGORICAL, CATEGORY_LABELS, CLEAN_CSV, FEATURES, NUMERIC_RANGES, RAW_CSV, TARGET


def clean_heart_data(input_csv=RAW_CSV, output_csv=CLEAN_CSV):
    df = pd.read_csv(input_csv)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing columns: {missing_cols}')
    df = df[FEATURES + [TARGET]]
    print('Raw shape:', df.shape)

    # 1. Exact duplicate rows
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f'Duplicates removed: {dupes}')

    # 2. Missing values: report only (imputed later, inside the pipeline)
    nans = df.isnull().sum()
    print('Missing values per column:\n', nans[nans > 0].to_string() if nans.any() else 'none')

    # 3. Categorical columns: nullable integer, unknown codes -> missing
    for col in CATEGORICAL:
        df[col] = df[col].round().astype('Int64')
        bad = ~df[col].isna() & ~df[col].isin(CATEGORY_LABELS[col].keys())
        if bad.any():
            print(f'{col}: {bad.sum()} unknown codes set to missing')
            df.loc[bad, col] = pd.NA

    # 4. Numeric columns: physiologically impossible values -> missing
    for col, (lo, hi) in NUMERIC_RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        if bad.any():
            print(f'{col}: {bad.sum()} out-of-range values set to missing')
            df.loc[bad, col] = float('nan')

    df[TARGET] = df[TARGET].astype(int)

    df.to_csv(output_csv, index=False)
    print(f'Clean shape: {df.shape}, disease rate: {df[TARGET].mean():.1%}')
    print(f'Saved to {output_csv}')
    return df


if __name__ == '__main__':
    clean_heart_data()
