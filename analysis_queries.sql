-- SQL Analysis Queries for heart_data table
-- Assumes target: 0=no disease, 1=disease
-- Run these in MySQL: USE heart_db;

-- 1. Disease vs No Disease count
SELECT 
    target,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM heart_data
GROUP BY target
ORDER BY target;

-- 2. Average cholesterol by disease
SELECT 
    target,
    ROUND(AVG(chol), 2) as avg_cholesterol
FROM heart_data
GROUP BY target
ORDER BY target;

-- 3. Gender vs Disease distribution (sex: 0=female, 1=male)
SELECT 
    sex,
    target,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY sex), 2) as pct_within_gender
FROM heart_data
GROUP BY sex, target
ORDER BY sex, target;

-- 4. Max heart rate comparison (avg thalach by disease)
SELECT 
    target,
    ROUND(AVG(thalach), 2) as avg_max_heart_rate,
    ROUND(STDDEV(thalach), 2) as std_heart_rate
FROM heart_data
GROUP BY target
ORDER BY target;

-- Bonus: Age distribution by disease
SELECT 
    target,
    ROUND(AVG(age), 2) as avg_age,
    MIN(age) as min_age,
    MAX(age) as max_age
FROM heart_data
GROUP BY target;
