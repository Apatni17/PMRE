import pandas as pd
import numpy as np
import os

# Define the directory for data
data_dir = 'PRME_App/data'
os.makedirs(data_dir, exist_ok=True)

# Define the path for the sample data file
sample_data_path = os.path.join(data_dir, 'sample_data.csv')

# Number of rows for each risk category
n_low = 300
n_medium = 350
n_high = 350

rows = []

# Low risk: low temp, low vibration, normal pressure, moderate runtime, few maintenance issues
for _ in range(n_low):
    row = {
        'Machine ID': f'L{np.random.randint(1000, 9999)}',
        'temperature': np.random.uniform(20, 40),
        'vibration': np.random.uniform(0.1, 1.5),
        'pressure': np.random.uniform(90, 120),
        'runtime': np.random.uniform(1000, 3000),
        'maintenance_history': np.random.randint(0, 2),
        'risk_score': np.random.uniform(10, 35)
    }
    rows.append(row)

# Medium risk: moderate temp/vibration, some pressure fluctuation, higher runtime, some maintenance
for _ in range(n_medium):
    row = {
        'Machine ID': f'M{np.random.randint(1000, 9999)}',
        'temperature': np.random.uniform(41, 70),
        'vibration': np.random.uniform(1.6, 3.0),
        'pressure': np.random.uniform(70, 130),
        'runtime': np.random.uniform(2000, 4000),
        'maintenance_history': np.random.randint(1, 4),
        'risk_score': np.random.uniform(40, 70)
    }
    rows.append(row)

# High risk: high temp, high vibration, abnormal pressure, long runtime, frequent maintenance
for _ in range(n_high):
    row = {
        'Machine ID': f'H{np.random.randint(1000, 9999)}',
        'temperature': np.random.uniform(71, 100),
        'vibration': np.random.uniform(3.1, 5.0),
        'pressure': np.random.uniform(50, 160),
        'runtime': np.random.uniform(3500, 6000),
        'maintenance_history': np.random.randint(3, 7),
        'risk_score': np.random.uniform(75, 100)
    }
    rows.append(row)

# Shuffle the rows for randomness
np.random.shuffle(rows)

# Create DataFrame and save
sample_df = pd.DataFrame(rows)
sample_df.to_csv(sample_data_path, index=False)

print(f"Sample data CSV created successfully at: {sample_data_path}")
print(sample_df['risk_score'].describe())
print(sample_df['risk_score'].value_counts(bins=[0,39,70,100])) 