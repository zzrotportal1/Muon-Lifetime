import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Initialize variables
params = {}
header_line = ''
data_lines = []
state = 0  # 0: reading parameters, 1: reading header, 2: reading data

# Open and read the file
with open('time difference.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue  # Skip empty lines
    if state == 0:
        if 'Ch0_time1' in line:
            # Found header line
            header_line = line
            state = 2
        else:
            # Process parameter lines
            items = line.split('\t')
            for item in items:
                key_value = item.strip().split('=')
                if len(key_value) == 2:
                    key, value = key_value
                    params[key.strip()] = value.strip()
    elif state == 2:
        # Reading data lines
        data_items = line.split('\t')
        data_lines.append(data_items)

# Process header line and data lines
header = header_line.split('\t')

# Handle missing values by padding shorter rows with empty strings
max_columns = len(header)
for i, data in enumerate(data_lines):
    if len(data) < max_columns:
        data_lines[i] += [''] * (max_columns - len(data))

# Create DataFrame
df = pd.DataFrame(data_lines, columns=header)

# Convert data columns to numeric values
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Display the extracted parameters and data
print("Parameters:")
for k, v in params.items():
    print(f"{k} = {v}")

print("\nData:")
print(df)

