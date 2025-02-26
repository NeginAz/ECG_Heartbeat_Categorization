import numpy as np
import pandas as pd

# Generate synthetic data (e.g., 10 samples with 187 features each)
num_samples = 10
num_features = 187

# Create random data (you can modify this to be more realistic if needed)
synthetic_data = np.random.normal(loc=0, scale=1, size=(num_samples, num_features))

# Convert to a DataFrame
df = pd.DataFrame(synthetic_data)

# Save to a CSV file without headers
df.to_csv('synthetic_ecg_data_no_headers.csv', index=False, header=False)
print('Synthetic test CSV file generated: synthetic_ecg_data_no_headers.csv')

