from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def GenerateSyntheticDataGmm(df, class_sample_map, n_components_map=None, random_state=42):
    """
    Generate synthetic ECG data using Gaussian Mixture Models (GMM) for multiple classes.
    
    Args:
    - df (pd.DataFrame): Combined dataframe with ECG data and labels.
    - class_sample_map (dict): Dictionary mapping class labels to target sample sizes.
    - n_components_map (dict, optional): Dictionary mapping class labels to GMM components. Defaults to 5 if not provided.
    - random_state (int): Seed for reproducibility.

    Returns:
    - pd.DataFrame: Synthetic feature data scaled to [0, 1].
    - pd.Series: Synthetic labels.
    """
    synthetic_data = []
    synthetic_labels = []
    
    #MinMaxScaler to scale synthetic samples to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))

    for label, target_samples in class_sample_map.items():
        # Extract data for the specified class
        class_data = df[df.iloc[:, -1] == label].iloc[:, :-1].values
        
        # Determine the number of GMM components for this class
        n_components = n_components_map.get(label, 5) if n_components_map else 1
        
        # Fit a Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(class_data)
        
        # Generate synthetic samples
        synthetic_samples = gmm.sample(target_samples)[0]
        
        # Scale the synthetic samples to the range [0, 1]
        synthetic_samples = scaler.fit_transform(synthetic_samples)
        
        # Create synthetic labels
        synthetic_data.append(synthetic_samples)
        synthetic_labels.append(np.full(target_samples, label))
    
    # Convert to DataFrame and Series
    X_synthetic = pd.DataFrame(np.vstack(synthetic_data))
    y_synthetic = pd.Series(np.hstack(synthetic_labels))
    
    return X_synthetic, y_synthetic



# from sklearn.mixture import GaussianMixture
# import pandas as pd
# import numpy as np

# def GenerateSyntheticDataGmm(df, class_sample_map, n_components_map=None, random_state=42):
#     """
#     Generate synthetic ECG data using Gaussian Mixture Models (GMM) for multiple classes.
    
#     Args:
#     - df (pd.DataFrame): Combined dataframe with ECG data and labels.
#     - class_sample_map (dict): Dictionary mapping class labels to target sample sizes.
#     - n_components_map (dict, optional): Dictionary mapping class labels to GMM components. Defaults to 5 if not provided.
#     - random_state (int): Seed for reproducibility.

#     Returns:
#     - pd.DataFrame: Synthetic feature data.
#     - pd.Series: Synthetic labels.
#     """
#     synthetic_data = []
#     synthetic_labels = []

#     for label, target_samples in class_sample_map.items():
#         # Extract data for the specified class
#         class_data = df[df.iloc[:, -1] == label].iloc[:, :-1].values
        
#         # Determine the number of GMM components for this class
#         n_components = n_components_map.get(label, 5) if n_components_map else 2
        
#         # Fit a Gaussian Mixture Model
#         gmm = GaussianMixture(n_components=n_components, random_state=random_state)
#         gmm.fit(class_data)
        
#         # Generate synthetic samples
#         synthetic_samples = gmm.sample(target_samples)[0]
        
#         # Create synthetic labels
#         synthetic_data.append(synthetic_samples)
#         synthetic_labels.append(np.full(target_samples, label))
    
#     # Convert to DataFrame and Series
#     X_synthetic = pd.DataFrame(np.vstack(synthetic_data))
#     y_synthetic = pd.Series(np.hstack(synthetic_labels))
    
#     return X_synthetic, y_synthetic
