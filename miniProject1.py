import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sat = pd.read_csv('autonomous_radar_telemetry_50k.csv')
print(f"Dataset Shape: {sat.shape}")


# Duplicate Checks

print(f"Total Missing Values: {sat.isnull().sum().sum()}")
duplicate_count = sat.duplicated().sum()
print(f"Total Duplicate Records: {duplicate_count}")

if duplicate_count > 0:
    sat.drop_duplicates(inplace=True)
    print("Duplicates removed in-place.")



# Outlier Treatment 

num_cols = sat.select_dtypes(include=[np.number]).columns.tolist()
if 'sensor_degraded_flag' in num_cols:
    num_cols.remove('sensor_degraded_flag')

for col in num_cols:
    Q1 = sat[col].quantile(0.25)
    Q3 = sat[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    sat[col] = sat[col].clip(lower_bound, upper_bound)



# Data Transformation & Distribution Normalization

num_cols = sat.select_dtypes(include=[np.number]).columns.tolist()
skewness_values = sat[num_cols].skew()

skewed_cols = skewness_values[skewness_values > 1.0].index.tolist()
print("Columns requiring log transformation based on skewness:", skewed_cols)

for col in skewed_cols:
    sat[f'log_{col}'] = np.log1p(sat[col])



# Normalization and Standardization 

scale_features = [
    'vehicle_speed_kmh', 'tx_power_dbm', 'snr_db', 
    'target_range_meters', 'sensor_temp_c', 'mcu_voltage_v'
]

scaler_minmax = MinMaxScaler()
scaler_std = StandardScaler()

sat[[f'{col}_norm' for col in scale_features]] = scaler_minmax.fit_transform(sat[scale_features])
sat[[f'{col}_std' for col in scale_features]] = scaler_std.fit_transform(sat[scale_features])



# Feature Engineering

# 1. Transmitter Power to MCU Voltage Efficiency Ratio
sat['power_to_voltage_ratio'] = sat['tx_power_dbm'] / sat['mcu_voltage_v']

# 2. Thermal Health Status Categorization
sat['temperature_category'] = pd.cut(
    sat['sensor_temp_c'],
    bins=[-np.inf, 40, 60, np.inf],
    labels=['Normal', 'High', 'Critical']
)

# 3. Signal-to-Noise Ratio (SNR) 
sat['snr_category'] = pd.cut(
    sat['snr_db'],
    bins=[-np.inf, 10, 20, np.inf],
    labels=['Low', 'Medium', 'High']
)


print(f"Final Dataset Shape: {sat.shape}")
print(f"Missing Values: {sat.isnull().sum().sum()}")
print(f"Duplicates: {sat.duplicated().sum()}")
