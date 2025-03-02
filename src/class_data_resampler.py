import numpy as np
import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt

class ClassDataResampler:
    """
    Resample class data to generate synthetic samples using resampling with replacement.
    
    Attributes:
        class_sample_map (dict): A dictionary mapping class labels to the target number of samples.
        random_state (int): Seed for reproducibility.
    """
    
    def __init__(self, class_sample_map, random_state=42):
        """
        Initialize the resampler with target samples for each class.
        
        Args:
            class_sample_map (dict): Mapping class labels to target sample sizes.
            random_state (int): Random state for reproducibility.
        """
        self.class_sample_map = class_sample_map
        self.random_state = random_state

    def resample_class_data(self, data, label):
        """
        Resample data for a specific class to generate synthetic samples.
        
        Args:
            data (np.array or pd.DataFrame): The input data of the specific class.
            label (int or float): The class label for resampling.
            
        Returns:
            pd.DataFrame: Resampled synthetic samples with labels.
        """
        # Determine the target number of samples for this class
        n_samples = self.class_sample_map.get(label, len(data))
        
        # Ensure the data is a numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Resample with replacement to generate more samples
        synthetic_samples = resample(
            data, 
            replace=True,                # Allow sampling with replacement
            n_samples=n_samples,         # Target number of samples
            random_state=self.random_state
        )
        
        # Create a DataFrame with labels in the last column
        synthetic_data_with_labels = pd.DataFrame(synthetic_samples)
        synthetic_data_with_labels['label'] = label
        
        return synthetic_data_with_labels

    def generate_resampled_data(self, df):
        """
        Generate synthetic data for all specified classes.
        
        Args:
            df (pd.DataFrame): The input DataFrame with the last column as labels.
        
        Returns:
            pd.DataFrame: Combined synthetic data with labels.
        """
        synthetic_data = []
        
        for label, target_samples in self.class_sample_map.items():
            print(f"Generating {target_samples} samples for class {label}...")
            
            # Extract data for the specified class
            class_data = df[df.iloc[:, -1] == label].iloc[:, :-1]
            
            # Generate synthetic samples for the class
            synthetic_class_data = self.resample_class_data(class_data, label)
            
            # Append to the overall synthetic data
            synthetic_data.append(synthetic_class_data)
        
        # Concatenate all synthetic data into a single DataFrame
        synthetic_data_combined = pd.concat(synthetic_data, ignore_index=True)
        
        return synthetic_data_combined

    def plot_samples(self, original_data, synthetic_data, label, n_samples=5):
        """
        Plot original vs. synthetic samples for a specific class.
        
        Args:
            original_data (pd.DataFrame): Original samples of the specific class.
            synthetic_data (pd.DataFrame): Synthetic samples generated for the class.
            label (int or float): Class label to visualize.
            n_samples (int): Number of samples to plot.
        """
        plt.figure(figsize=(10, 6))
        
        original_data = original_data.values if isinstance(original_data, pd.DataFrame) else original_data
        synthetic_data = synthetic_data.iloc[:, :-1].values  # Exclude labels column
        
        for i in range(n_samples):
            plt.plot(original_data[i], label=f'Original {i}', alpha=0.7)
            plt.plot(synthetic_data[i], label=f'Synthetic {i}', linestyle='--', alpha=0.8)

        plt.title(f'Original vs. Resampled Synthetic Samples for Class {label}')
        plt.xlabel('Feature Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'Original vs. Resampled Synthetic Samples for Class {label}.png', bbox_inches='tight', dpi=300)
        plt.show()


