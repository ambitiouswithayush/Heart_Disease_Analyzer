#!/bin/bash
# One-command setup: install dependencies, run the pipeline and the tests
set -e
python3 -m pip install -r requirements.txt
python3 run_pipeline.py
python3 -m pytest -q
echo "Done. Start the app with: python3 app.py  ->  http://127.0.0.1:5000"
