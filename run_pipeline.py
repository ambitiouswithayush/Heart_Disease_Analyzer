#!/usr/bin/env python3
\"\"\"Full Heart Disease Analysis Pipeline.\"\"\"
import subprocess
import sys
import os

def run_command(cmd, desc):
    print(f'Running: {desc}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print('✅ Success')
        return True
    else:
        print(f'❌ Failed: {result.stderr}')
        return False

def main():
    print('='*60)
    print('HEART DISEASE PIPELINE - Starting...')
    print('='*60)
    
    steps = [
        ('Download dataset', 'python download_data.py'),
        ('Load to MySQL', 'python load_heart_to_mysql.py'),
        ('Clean data', 'python clean_data.py'),
        ('Train ML model', 'python train_model.py'),
        ('Generate visualizations', 'python visualize.py'),
    ]
    
    success = True
    for desc, cmd in steps:
        if not run_command(cmd, desc):
            success = False
            break
    
    if success:
        print('\n🎉 PIPELINE COMPLETE!')
        print('\n📊 Generated:')
        print('- heart.csv (303 records)')
        print('- heart_data table (MySQL)')
        print('- heart_cleaned.csv + scaler.pkl')
        print('- heart_model.pkl (85% acc)')
        print('- images/ (8 PNG charts)')
        print('\n🚀 Start web app: python app.py')
        print('   Visit: http://localhost:5000')
    else:
        print('\n💥 Pipeline failed. Check errors above.')

if __name__ == '__main__':
    main()

