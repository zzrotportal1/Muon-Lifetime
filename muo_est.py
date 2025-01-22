import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = 'MUO_temp_filtered_pulse_data.csv'

with open(filename, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip().startswith('Ch0_time1'):
        header_line = i
        break
else:
    print("Column headers not found in the file.")
    exit(1)

# Read the data into a DataFrame, starting from the header line
# Use 'sep=',\s+' to handle any amount of whitespace
df = pd.read_csv(filename, skiprows=header_line, sep=',', engine='python')

# Strip any whitespace from column headers
df.columns = df.columns.str.strip()


# Proceed with the calculations
# Calculate the time difference dt = Ch0_time2 - Ch0_time1
df['dt'] = df['Ch0_time2'] - df['Ch0_time1 (s)']

# Remove any entries where dt is not positive
df = df[df['dt'] > 0]

# Convert dt to microseconds for convenience
df['dt_us'] = df['dt'] * 1e6

# Check how many data points we have
print(f"Number of data points: {len(df)}")

# Plot histogram of dt_us
plt.figure(figsize=(8,6))
counts, bin_edges, _ = plt.hist(df['dt_us'], bins=99, range=(0,40), histtype='step', label='Data')

plt.xlabel('Time difference dt (μs)')
plt.ylabel('Counts')
plt.title('Muon Decay Time Differences')
plt.legend()
plt.show()
# Prepare data for fitting using all bins
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Use all bins without any mask
x_fit = bin_centers
y_fit = counts

# Remove bins with zero counts to avoid issues in the fit
nonzero = y_fit > 0
x_fit = x_fit[nonzero]
y_fit = y_fit[nonzero]

# Check if y_fit is empty
if len(y_fit) == 0:
    print("No data in the fitting range. Adjust the fitting range or check the data.")
    exit(1)

# Define the exponential decay function
def exponential(t, N0, tau):
    return N0 * np.exp(-t / tau)

# Initial guesses for N0 and tau
N0_guess = y_fit[0]
tau_guess = 2.2  # Initial guess for muon lifetime in μs

# Fit the data to the exponential decay function
popt, pcov = curve_fit(exponential, x_fit, y_fit, p0=[N0_guess, tau_guess])

N0_fit, tau_fit = popt
tau_err = np.sqrt(pcov[1,1])

print(f"Estimated muon lifetime: {tau_fit:.2f} ± {tau_err:.2f} μs")

plt.figure(figsize=(8,6))
plt.hist(df['dt_us'], bins=99, range=(0,40), histtype='step', label='Data')

t_fit = np.linspace(0, 40, 400)
y_fit_curve = exponential(t_fit, N0_fit, tau_fit)

plt.plot(t_fit, y_fit_curve, 'r-', label=f'Fit: τ = {tau_fit:.2f} ± {tau_err:.2f} μs')

plt.xlabel('Time difference dt (μs)')
plt.ylabel('Counts')
plt.title('Muon Decay Time Differences with Exponential Fit')
plt.legend()
plt.show()
