# Insights

All figures come from `heart.csv` (303 patients) and `artifacts/metrics.json`.

## 1. Who has heart disease?
- **Prevalence:** 139 / 303 = 45.9%. Classes are roughly balanced, so accuracy is meaningful, but recall still matters most.
- **Sex:** 55% of men vs 26% of women.
- **Chest pain:** *asymptomatic* patients (type 4) have a 73% disease rate, much higher than typical angina (30%). Silent disease is common in this referral population.
- **Blocked vessels (`ca`):** 26% → 68% → 82% → 85% for 0 → 3 vessels, a near-monotonic dose-response.
- **Exercise angina:** 77% if present vs 31% if absent.
- **Thallium test:** reversible defect 76%, fixed defect 67%, normal 22%.

## 2. What doesn't matter much?
- **Cholesterol** (252 vs 243 mg/dl) and **fasting blood sugar** add almost nothing once other features are known (permutation importance ≈ 0).
- **Age** matters on its own (57 vs 53 years), but adds little once vessels and heart rate are known, because they already capture age-related decline.

## 3. Model
- Logistic Regression (C = 0.1) had the best CV ROC-AUC, 0.911 ± 0.018, beating Random Forest (0.896) and Gradient Boosting (0.873).
- Largest odds multipliers from the coefficients (numeric features per +1 standard deviation; categories relative to the average category): blocked vessels ×2.0, asymptomatic chest pain ×1.7, reversible thal defect ×1.65, male ×1.5; higher max heart rate ×0.73 (protective).
- At the tuned threshold (0.31) the model missed **0 of 28** sick patients in the test set, at the cost of 11 false alarms out of 33 healthy patients.

## 4. Recommendations
1. Use the model as a **triage aid** to prioritise patients for further cardiac testing, not to diagnose.
2. Pay attention to asymptomatic patients with exercise-induced ST depression and a low max heart rate.
3. Validate on another hospital's data before trusting the numbers anywhere else.
