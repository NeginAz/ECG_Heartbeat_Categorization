import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture




def stratified_sample(df: pd.DataFrame, percentage: float, label_col_index: int = -1) -> (pd.DataFrame, pd.DataFrame):
    """
    Perform stratified sampling on a DataFrame without headers, ensuring equal representation of each label.

    Args:
        df (pd.DataFrame): The input DataFrame (no header assumed).
        percentage (float): The percentage of samples to select.
        label_col_index (int): The index of the label column (default is the last column).

    Returns:
        (pd.DataFrame, pd.DataFrame): Sampled and remaining DataFrames.
    """
    # Calculate the total number of samples needed
    total_samples = int(len(df) * (percentage / 100))
    
    # Determine the unique labels and samples per label
    labels = df.iloc[:, label_col_index].unique()  # Specify label column by index
    samples_per_label = total_samples // len(labels)
    
    # Perform stratified sampling
    sampled_dfs = []
    for label in labels:
        label_df = df[df.iloc[:, label_col_index] == label]
        
        # Handle cases where samples_per_label might be greater than available samples
        if len(label_df) < samples_per_label:
            raise ValueError(f"Not enough samples for label {label} to fulfill stratified sampling.")
        
        sampled_label_df = label_df.sample(n=samples_per_label, random_state=42)
        sampled_dfs.append(sampled_label_df)
    
    # Concatenate sampled data and shuffle
    sampled_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Get the remaining data by dropping the sampled indices
    remaining_df = df.drop(sampled_df.index).reset_index(drop=True)
    
    return sampled_df, remaining_df


def compute_bic_for_one_class(df, class_label, label_column, max_components=10):
    """
    Computes BIC for a specific class in the dataset using GMM.

    Args:
        df (pd.DataFrame): Dataframe containing features and labels.
        class_label (int or float): The specific class to analyze.
        label_column (str): Name of the label column.
        max_components (int): Maximum number of GMM components to test.

    Returns:
        list: BIC scores for different GMM component numbers.
    """
    # Extract data for the specified class
    class_data = df[df[label_column] == class_label].iloc[:, :-1].values  # Exclude label column

    bic_values = []
    components_range = range(1, max_components + 1)

    for n in components_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(class_data)
        bic_values.append(gmm.bic(class_data))  # Store BIC score

    # Plot BIC scores
    plt.plot(components_range, bic_values, marker='o', linestyle='-', label=f'Class {class_label}')
    plt.xlabel("Number of Components")
    plt.ylabel("BIC Score")
    plt.title(f"BIC Scores for Class {class_label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"BIC Scores for Class {class_label}.png", bbox_inches='tight', dpi=300)
    plt.show()

    return bic_values




