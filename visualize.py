"""Exploratory data analysis charts (on raw, unscaled values)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CATEGORY_LABELS, CLEAN_CSV, IMAGES_DIR, TARGET

sns.set_theme(style='whitegrid')
PALETTE = {'No disease': '#4C9BE8', 'Disease': '#E8604C'}


def save(fig, name):
    fig.savefig(os.path.join(IMAGES_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_visualizations(csv_path=CLEAN_CSV):
    df = pd.read_csv(csv_path)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    df['Status'] = df[TARGET].map({0: 'No disease', 1: 'Disease'})
    print(f'Visualizing {len(df)} records')

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df['Status'].value_counts()
    ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f'{v} ({v / len(df):.0%})', ha='center')
    ax.set_title('Class balance')
    ax.set_ylabel('Patients')
    save(fig, '01_class_balance.png')

    # 2-4. Numeric features by disease status
    for fname, col, title in [('02_age_by_status.png', 'age', 'Age (years)'),
                              ('03_max_hr_by_status.png', 'thalach', 'Maximum heart rate'),
                              ('04_cholesterol_by_status.png', 'chol', 'Cholesterol (mg/dl)')]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='Status', y=col, hue='Status', palette=PALETTE, legend=False, ax=ax)
        ax.set_title(f'{title} by disease status')
        ax.set_xlabel('')
        ax.set_ylabel(title)
        save(fig, fname)

    # 5. Disease rate by each categorical feature
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, ['sex', 'cp', 'exang', 'slope', 'thal', 'ca']):
        rate = df.groupby(col)[TARGET].agg(['mean', 'count'])
        labels = [CATEGORY_LABELS.get(col, {}).get(int(i), str(int(i))) for i in rate.index]
        ax.bar(labels, rate['mean'], color='#E8604C')
        for i, (m, n) in enumerate(zip(rate['mean'], rate['count'])):
            ax.text(i, m + 0.02, f'{m:.0%}\nn={n}', ha='center', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(f'Disease rate by {col}')
        ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    save(fig, '05_disease_rate_by_category.png')

    # 6. Correlation matrix (numeric features + target)
    corr = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation matrix')
    save(fig, '06_correlation.png')

    # 7. Age vs max heart rate
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x='age', y='thalach', hue='Status', palette=PALETTE, ax=ax)
    ax.set_title('Age vs maximum heart rate')
    save(fig, '07_age_vs_max_hr.png')

    # 8. ST depression distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x='oldpeak', hue='Status', palette=PALETTE, bins=20, multiple='layer', ax=ax)
    ax.set_title('ST depression (oldpeak) by disease status')
    save(fig, '08_oldpeak_distribution.png')

    print(f'8 EDA charts saved to {IMAGES_DIR}/ (model charts 09-12 come from train_model.py)')


if __name__ == '__main__':
    create_visualizations()
