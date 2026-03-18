#!/bin/bash
# Heart Disease Project Setup & Run

echo \"🚀 Setting up Heart Disease Analysis Project...\"

# Install dependencies
echo \"📦 Installing Python dependencies...\"
pip install -r requirements.txt

# Download data and run pipeline
echo \"⬇️  Downloading data & running pipeline...\"
python download_data.py
python run_pipeline.py

echo \"\n✅ Setup complete!\"
echo \"🌐 Start web app: python app.py\"
echo \"📱 Visit: http://localhost:5000/\"
echo \"🔮 ML Predict: http://localhost:5000/predict\"
echo \"📊 MySQL: mysql -u root -p heart_db < analysis_queries.sql\"

echo \"\nRun 'python app.py' to start the interactive dashboard!\"

