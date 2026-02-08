import pandas as pd
import numpy as np
import os

# Make sure the output folder exists
os.makedirs('data', exist_ok=True)

# Fixed seed so the same dataset can be regenerated
np.random.seed(42)

# Basic settings
n_samples = 5000

# Generate features
# Age: 20 to 80
age = np.random.randint(20, 80, n_samples)

# BMI: 18.5 to 40.0
bmi = np.random.uniform(18.5, 40.0, n_samples)

# Blood Pressure (Systolic): 90 to 180
bp = np.random.randint(90, 180, n_samples)

# Cholesterol: 150 to 300
cholesterol = np.random.randint(150, 300, n_samples)

# Glucose: 70 to 200
glucose = np.random.randint(70, 200, n_samples)

# Gender: 0 (Female), 1 (Male)
gender = np.random.randint(0, 2, n_samples)

# Assemble into a table
df = pd.DataFrame({
    'Age': age,
    'BMI': bmi,
    'BloodPressure': bp,
    'Cholesterol': cholesterol,
    'Glucose': glucose,
    'Gender': gender
})

# Build a simple risk score to simulate labels (not clinical logic)
# Base risk score
risk_score = (
    (df['Age'] - 50) / 30 * 2 + 
    (df['BMI'] - 25) / 10 * 1.5 + 
    (df['BloodPressure'] - 120) / 20 * 1.5 + 
    (df['Cholesterol'] - 200) / 40 * 1.0 +
    (df['Glucose'] - 100) / 30 * 2.0
)

# Add some noise so it's not too clean
risk_score += np.random.normal(0, 1.5, n_samples)

# Define classes based on thresholds
# 0: Healthy, 1: Diabetes risk (risk > 1.5), 2: Heart disease risk (risk > 3.0)
# This intentionally creates imbalance for the training demo.

conditions = [
    (risk_score > 3.0),
    (risk_score > 1.5)
]
choices = [2, 1] # 2: Heart Disease, 1: Diabetes
df['Disease'] = np.select(conditions, choices, default=0)

# Quick sanity check
print("Class distribution:\n", df['Disease'].value_counts())

# Save to CSV
df.to_csv('data/medical_data.csv', index=False)
print("Saved dataset to data/medical_data.csv")
