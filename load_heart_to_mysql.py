"""Load the raw CSV into MySQL for SQL analysis (optional step).

Connection settings come from environment variables:
MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD (defaults: localhost, root, empty).
"""
import os

import mysql.connector
import pandas as pd

from config import FEATURES, RAW_CSV, TARGET

COLUMNS = FEATURES + [TARGET]


def connect():
    return mysql.connector.connect(
        host=os.environ.get('MYSQL_HOST', 'localhost'),
        user=os.environ.get('MYSQL_USER', 'root'),
        password=os.environ.get('MYSQL_PASSWORD', ''),
    )


def main():
    df = pd.read_csv(RAW_CSV)[COLUMNS]
    # Missing values become SQL NULL
    rows = [tuple(None if pd.isna(v) else float(v) for v in r) for r in df.itertuples(index=False)]

    conn = connect()
    cursor = conn.cursor()
    try:
        cursor.execute('CREATE DATABASE IF NOT EXISTS heart_db')
        cursor.execute('USE heart_db')
        cursor.execute('DROP TABLE IF EXISTS heart_data')
        cursor.execute('''
            CREATE TABLE heart_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                age INT, sex INT, cp INT, trestbps INT, chol INT, fbs INT, restecg INT,
                thalach INT, exang INT, oldpeak FLOAT, slope INT, ca INT NULL, thal INT NULL,
                target INT
            )''')
        placeholders = ', '.join(['%s'] * len(COLUMNS))
        cursor.executemany(f'INSERT INTO heart_data ({", ".join(COLUMNS)}) VALUES ({placeholders})', rows)
        conn.commit()
        print(f'Loaded {len(rows)} rows into heart_db.heart_data')
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    main()
