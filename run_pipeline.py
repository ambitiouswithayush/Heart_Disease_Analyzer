#!/usr/bin/env python3
"""Run the full pipeline: download -> (MySQL) -> clean -> EDA -> train."""
import subprocess
import sys

STEPS = [
    ('Download dataset', 'download_data.py', True),
    ('Load into MySQL (optional)', 'load_heart_to_mysql.py', False),
    ('Validate and clean data', 'clean_data.py', True),
    ('EDA charts', 'visualize.py', True),
    ('Train, compare and evaluate models', 'train_model.py', True),
]


def main():
    for desc, script, required in STEPS:
        print(f'\n=== {desc} ===')
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            if required:
                sys.exit(f'Step failed: {desc}')
            print(f'Skipped: {desc} (MySQL not available; the rest of the pipeline does not need it)')
    print('\nPipeline complete. Start the web app with: python app.py  ->  http://127.0.0.1:5000')


if __name__ == '__main__':
    main()
