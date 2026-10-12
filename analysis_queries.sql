-- SQL analysis on heart_db.heart_data (target: 0 = no disease, 1 = disease)
USE heart_db;

-- 1. Class balance with percentage (window function over the grouped result)
SELECT target,
       COUNT(*) AS patients,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM heart_data
GROUP BY target;

-- 2. Disease rate by sex
SELECT CASE sex WHEN 1 THEN 'Male' ELSE 'Female' END AS sex,
       COUNT(*) AS patients,
       ROUND(AVG(target) * 100, 1) AS disease_rate_pct
FROM heart_data
GROUP BY sex;

-- 3. Disease rate by chest pain type (4 = asymptomatic has the highest rate)
SELECT cp, COUNT(*) AS patients, ROUND(AVG(target) * 100, 1) AS disease_rate_pct
FROM heart_data
GROUP BY cp
ORDER BY disease_rate_pct DESC;

-- 4. Disease rate by age band
SELECT CASE WHEN age < 45 THEN '<45'
            WHEN age < 55 THEN '45-54'
            WHEN age < 65 THEN '55-64'
            ELSE '65+' END AS age_band,
       COUNT(*) AS patients,
       ROUND(AVG(target) * 100, 1) AS disease_rate_pct
FROM heart_data
GROUP BY age_band
ORDER BY MIN(age);

-- 5. Clinical averages by disease status
SELECT target,
       ROUND(AVG(age), 1) AS avg_age,
       ROUND(AVG(chol), 1) AS avg_chol,
       ROUND(AVG(thalach), 1) AS avg_max_hr,
       ROUND(STDDEV(thalach), 1) AS std_max_hr,
       ROUND(AVG(oldpeak), 2) AS avg_st_depression
FROM heart_data
GROUP BY target;

-- 6. Missing values check (ca and thal have NULLs in the raw data)
SELECT SUM(ca IS NULL) AS missing_ca, SUM(thal IS NULL) AS missing_thal
FROM heart_data;

-- 7. CTE + ranking: highest-risk subgroups (sex x chest pain) with at least 10 patients
WITH groups AS (
    SELECT sex, cp, COUNT(*) AS patients, AVG(target) AS disease_rate
    FROM heart_data
    GROUP BY sex, cp
    HAVING COUNT(*) >= 10
)
SELECT sex, cp, patients,
       ROUND(disease_rate * 100, 1) AS disease_rate_pct,
       RANK() OVER (ORDER BY disease_rate DESC) AS risk_rank
FROM groups;
