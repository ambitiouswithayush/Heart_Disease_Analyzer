import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

plt.style.use('default')
sns.set_palette("husl")

def create_visualizations(csv_path='heart_cleaned.csv'):
    if not os.path.exists(csv_path):
        print(f'{csv_path} not found. Run clean_data.py first.')
        return
    
    df = pd.read_csv(csv_path)
    print(f'Visualizing {len(df)} records')
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # 1. Disease Distribution (Bar + Pie)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    disease_counts = df['target'].value_counts()
    disease_counts.plot(kind='bar', ax=ax1, title='Disease Distribution')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
    
    disease_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', title='Disease %')
    ax2.tick_params(axis='x', rotation=0)
    plt.savefig('images/01_disease_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Age vs Target (Violin)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='target', y='age', palette=['skyblue', 'salmon'])
    plt.title('Age Distribution by Disease Status')
    plt.xlabel('Target (0=No, 1=Yes)')
    plt.savefig('images/02_age_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cholesterol by Target (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='target', y='chol', palette=['lightgreen', 'orange'])
    plt.title('Cholesterol Distribution by Disease')
    plt.xlabel('Target (0=No, 1=Yes)')
    plt.savefig('images/03_cholesterol_box.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Gender vs Target (Crosstab heatmap)
    crosstab = pd.crosstab(df['sex'], df['target'], normalize='index') * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(crosstab, annot=True, cmap='RdYlBu_r', fmt='.1f')
    plt.title('Gender vs Disease (%)')
    plt.ylabel('Sex (0=F, 1=M)')
    plt.savefig('images/04_gender_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation Heatmap (Numerical)
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    corr = df[num_cols + ['target']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('images/05_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Chest Pain Type Distribution
    plt.figure(figsize=(10, 6))
    cp_target = pd.crosstab(df['cp'], df['target'])
    cp_target.plot(kind='bar', stacked=True)
    plt.title('Chest Pain Type vs Disease')
    plt.xlabel('CP Type (0-3)')
    plt.ylabel('Count')
    plt.legend(title='Target')
    plt.savefig('images/06_cp_stackedbar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Max Heart Rate Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='age', y='thalach', hue='target', size='chol', 
                    sizes=(20, 200), palette=['blue', 'red'])
    plt.title('Age vs Max Heart Rate (colored by disease, sized by cholesterol)')
    plt.savefig('images/07_scatter_thalach.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Feature Importance (if model available)
    try:
        from joblib import load
        model = load('heart_model.pkl')
        importances = pd.DataFrame({
            'feature': df.columns[:-1],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importances.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importances (RandomForest)')
        plt.savefig('images/08_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('Feature importance chart created!')
    except:
        print('Model not found - skipping feature importance')
    
    print('✅ 8 visualizations saved to images/ folder!')
    print('Files: 01_disease_distribution.png ... 08_feature_importance.png')

if __name__ == '__main__':
    create_visualizations()
