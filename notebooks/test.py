# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#%%  

#load each dataset
df_normal = pd.read_csv("../Data/ptbdb_normal.csv" , header=None)
df_abnormal = pd.read_csv("../Data/ptbdb_abnormal.csv", header=None)
df_train = pd.read_csv("../Data/mitbih_train.csv", header=None)
df_test = pd.read_csv("../Data/mitbih_test.csv", header=None)



#%%
#check the dimensions of each dataset
print(df_normal.shape)
print(df_abnormal.shape)
print(df_train.shape)
print(df_test.shape)

#%% 

#check the first few rows of each dataset
print(df_normal.head())
print(df_abnormal.head())
print(df_train.head())
print(df_test.head())

#%%
# Distribution of labels
print(df_train.iloc[:, -1].value_counts())
print(df_test.iloc[:,-1].value_counts())

print(df_normal.iloc[:, -1].value_counts())
print(df_abnormal.iloc[:,-1].value_counts())

#%%
# Map PTB labels to align with MIT-BIH labels
df_normal.iloc[:, -1] = 0  # Normal -> 'N' (0)
df_abnormal.iloc[:, -1] = 4  # Abnormal -> 'Q' (4)

#%% Concatenate the datasets
df_ptb_combined = pd.concat([df_normal, df_abnormal])

df_mit_combined = pd.concat([df_train, df_test])


print(df_ptb_combined.iloc[:, -1].unique())
print(df_mit_combined.iloc[:,-1].unique())

#Concatenate PTB with MIT-BIH datasets
#df_combined = pd.concat([df_mit_combined, df_ptb_combined], axis=0)

df_combined = df_mit_combined

#%% Split the Dataset:

from sklearn.model_selection import train_test_split

# Separate features and labels
X = df_combined.iloc[:, :-1]
y = df_combined.iloc[:, -1]

# Split into training, test, and holdout sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Split the remaining data into test and holdout sets (50% each)
X_test, X_holdout, y_test, y_holdout = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Holdout set size: {X_holdout.shape[0]}")

#%% Repplace 0 (missing values) with Nan

import numpy as np

# Replace zeros with NaN in the training set only
X_train_no_zeros = X_train.replace(0, np.nan)

# Check for NaN values in the training set
missing_values = X_train_no_zeros.isnull().sum().sum()
print(f"Total missing values in the training set (NaN): {missing_values}")


#%% Replace Nan with the row-wise mean

row_means = X_train_no_zeros.mean(axis=1, numeric_only=True)


X_train_imputed = X_train_no_zeros.where(~ X_train_no_zeros.isna(), row_means, axis=0)

# Replace NaN values with the mean of each row
# mean_row = X_train_no_zeros.mean(axis=1)
# X_train_imputed = X_train_no_zeros.T.fillna(mean_row).T

# Verify if there are any remaining missing values
#print("Remaining missing values in the training set after row-wise mean imputation:")
#print(X_train_imputed.isnull().sum().sum())

#%%
# # Check if any NaN values remain
print("Total remaining NaN values in the dataset:")
print(X_train_imputed.isnull().sum().sum())


#%%

# import numpy as np
# import pandas as pd

# import numpy as np
# import pandas as pd

# # Combine features and labels for easier processing
# df_train_imputation = pd.concat([X_train_no_zeros, y_train], axis=1)

# # Initialize a dictionary to store class-wise means
# class_means = {}

# # Calculate class-wise mean for each feature, ensuring no NaN in means
# for class_label in df_train_imputation.iloc[:, -1].unique():
#     class_data = df_train_imputation[df_train_imputation.iloc[:, -1] == class_label]
    
#     # Fill class-specific means and handle potential NaN means
#     class_means[class_label] = class_data.iloc[:, :-1].apply(lambda col: col.mean() if col.mean() is not np.nan else 0)

# print("Class-wise means calculated for imputation:")
# print({k: v.mean() for k, v in class_means.items()})  # Display average of means per class


# # Create a copy of the training data for imputation
# X_train_imputed = X_train_no_zeros.copy()

# # Apply class-specific mean imputation
# for class_label, mean_values in class_means.items():
#     # Select rows for the current class
#     class_mask = (df_train_imputation.iloc[:, -1] == class_label)
    
#     # Fill NaN values with the class-specific mean
#     X_train_imputed.loc[class_mask] = X_train_imputed.loc[class_mask].fillna(mean_values)

# # Apply a global mean for any remaining NaNs (safety net)
# global_mean = X_train_imputed.mean()
# X_train_imputed = X_train_imputed.fillna(global_mean)

# # Check for remaining missing values
# remaining_nans = X_train_imputed.isnull().sum().sum()
# print("Remaining missing values after class-wise and global mean imputation:", remaining_nans)

# # If still NaNs, forcefully fill with 0 (final safety measure)
# if remaining_nans > 0:
#     X_train_imputed = X_train_imputed.fillna(0)
#     print("Forced fill of remaining NaNs with 0.")


# # Check if any NaN values remain
# print("Total remaining NaN values in the dataset:")
# print(X_train_imputed.isnull().sum().sum())

# # Quick check for any problematic columns
# print("Columns with NaN values:")
# print(X_train_imputed.columns[X_train_imputed.isnull().any()])

#%% Identify the Classes with Fewer Samples:

# Calculate the class distribution
class_distribution = y_train.value_counts().sort_values()

# Identify the two minority classes
minority_classes = class_distribution.index[:2]
print(f"Minority classes: {minority_classes.tolist()}")
print("Class distribution in the training set:")
print(class_distribution)




#%%
# import sys
# import os

# sys.path.append(os.path.abspath('../src'))
# from dimensionality_reduction import DimensionalityReductionVisualization 

# # Prepare the data with labels as a DataFrame
# data_with_labels = X_train_imputed.copy()
# data_with_labels['label'] = y_train

# # Initialize the visualizer
# visualizer = DimensionalityReductionVisualization(n_components_pca=4, label_column='label')

# # Plot PCA and t-SNE for a specific class
# visualizer.plot_pca(data_with_labels, target_class=3)
# #visualizer.plot_tsne(data_with_labels, target_class=3)





#%%  Generate Synthetic Data with Different Samples per Class:
# Define target sample sizes for each minority class
class_sample_map = {
    3.0: 12400,  # Generate 5000 samples for class 3.0
    1.0: 11000,  # Generate 7000 samples for class 1.0
    2.0: 7900   # Generate 6000 samples for class 2.0
}

# Optionally, define custom GMM components for each class
n_components_map = {
    3.0: 3,  # Use 3 components for class 3.0
    1.0: 5,  # Use 5 components for class 1.0
    2.0: 4   # Use 4 components for class 2.0
}


# Import the GMM data augmentation function
import sys
import os

sys.path.append(os.path.abspath('../src'))


from generate_synthetic_data_gmm import GenerateSyntheticDataGmm

# Generate synthetic data for the specified classes
X_synthetic, y_synthetic = GenerateSyntheticDataGmm(
    pd.concat([X_train_imputed, y_train], axis=1),
    class_sample_map=class_sample_map,
    n_components_map=n_components_map
)
print(f"Generated {X_synthetic.shape[0]} synthetic samples for classes {list(class_sample_map.keys())}")


#%% 

import pandas as pd

# Concatenate the synthetic samples with the original training data
X_train_augmented = pd.concat([X_train_imputed, X_synthetic], axis=0, ignore_index=True)
y_train_augmented = pd.concat([y_train, y_synthetic], axis=0, ignore_index=True)

print(f"New training set size after GMM augmentation: {X_train_augmented.shape[0]}")
print(f"Number of labels in augmented set: {y_train_augmented.shape[0]}")

#%% Prepare Data
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from cnn_lstm_classifier import CNNLSTMClassifier  # Assuming the model class is in the src directory

# Ensure the data is in numpy array format for Keras
X_data = np.expand_dims(X_train_augmented.values, axis=-1)  # Add channel dimension for CNN
y_data = y_train_augmented.values

print(f"Input shape for model: {X_data.shape}")
print(f"Labels shape: {y_data.shape}")

#%%
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
from data_augmentation import DataAugmentation  # Assuming the class is in the src folder
from cnn_lstm_classifier import CNNLSTMClassifier  # Assuming the model class is in the src directory

# Initialize Data Augmentation class
augmenter = DataAugmentation(
    shift_max=5, 
    noise_level=0.01, 
    scale_range=(0.9, 1.1), 
    crop_size=50
)

# Define the augmentation factor (e.g., 0.5, 1, 2, etc.)
augmentation_factor = 2  # You can adjust this value as needed

# 🆕 Apply data augmentation to the entire dataset before splitting
X_augmented = augmenter.augment_batch(X_data, augmentation_factor=augmentation_factor)

# Calculate the correct number of labels for the augmented samples
num_original_samples = len(y_data)
num_augmented_samples = X_augmented.shape[0]

# Generate the correct number of labels for the augmented samples
y_augmented = np.repeat(y_data, np.ceil(num_augmented_samples / num_original_samples).astype(int))[:num_augmented_samples]

# Combine original and augmented data
X_combined = np.vstack((X_data, X_augmented))
y_combined = np.concatenate((y_data, y_augmented))

# Shuffle the combined dataset
shuffle_indices = np.random.permutation(X_combined.shape[0])
X_combined = X_combined[shuffle_indices]
y_combined = y_combined[shuffle_indices]

print(f"Combined dataset size: {X_combined.shape[0]}")

# Initialize Stratified K-Fold with 5 splits
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# To store evaluation metrics
accuracy_scores = []
classification_reports = []
confusion_matrices = []

fold_number = 1

for train_index, val_index in kf.split(X_combined, y_combined):
    print(f"\nStarting training for Fold {fold_number}...")
    
    # Split the data into training and validation sets
    X_train_fold, X_val_fold = X_combined[train_index], X_combined[val_index]
    y_train_fold, y_val_fold = y_combined[train_index], y_combined[val_index]
    
    print(f"Training set size: {X_train_fold.shape[0]}")
    print(f"Validation set size: {X_val_fold.shape[0]}")
    
    # Initialize the CNN + LSTM model
    model = CNNLSTMClassifier(
        input_shape=(187, 1),  # The input shape is already correctly set
        num_classes=len(np.unique(y_combined)),
        learning_rate=1e-3,
        batch_size=32,
        epochs=50
    )
    
    # Train the model with the combined data
    model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    
    # Evaluate the model using the class's evaluate method
    print(f"\nEvaluating Fold {fold_number}...")
    model.evaluate(X_val_fold, y_val_fold)
    
    # Load the best model to make predictions
    best_model = keras.models.load_model('best_model.h5')
    
    # Predict on the validation set using the best model
    y_val_pred = best_model.predict(X_val_fold)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val_fold, y_val_pred_classes)
    accuracy_scores.append(accuracy)
    
    print(f"Accuracy for Fold {fold_number}: {accuracy:.4f}")
    
    # Generate classification report and confusion matrix
    classification_reports.append(classification_report(y_val_fold, y_val_pred_classes, digits=4))
    confusion_matrices.append(confusion_matrix(y_val_fold, y_val_pred_classes))
    
    fold_number += 1

#%%
model.plot_history()

#%% evlauate the model with the test set

# Ensure the test data has the correct shape
X_test = np.expand_dims(X_test.values, axis=-1)  # Add channel dimension if not already added
print(f"Test set shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")


from tensorflow import keras

# Load the best model weights
best_model = keras.models.load_model('best_model.h5')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Predict on the test set
y_test_pred = best_model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Generate classification report
print("Classification Report on Test Set:")
print(classification_report(y_test, y_test_pred_classes, digits=4))

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred_classes)
print(f"Accuracy on Test Set: {test_accuracy:.4f}")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
plt.show()

#%% Hold-out set

import numpy as np
from data_augmentation import DataAugmentation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

# Initialize the Data Augmentation class for noise injection
augmenter = DataAugmentation(
    shift_max=0,       # No time shift
    noise_level=0.1,   # Introduce significant noise
    scale_range=(1.0, 1.0), # No scaling
    crop_size=0,     # Safe cropping size
    expected_length=187
)

# 🆕 Prepare holdout data with the channel dimension for augmentation
X_holdout_prepared = np.expand_dims(X_holdout.values, axis=-1)  # Shape should be (samples, 187, 1)
y_holdout = y_holdout.values

#%%
print(f"Prepared holdout set shape for augmentation: {X_holdout_prepared.shape}")

# Apply noise to the holdout set with an augmentation factor of 1 (100% of samples)
X_holdout_noisy = augmenter.augment_batch(X_holdout_prepared, augmentation_factor=1.0)

# 🆕 Set the labels for the augmented noisy holdout set
num_original_samples = len(y_holdout)
num_augmented_samples = X_holdout_noisy.shape[0]

# Generate the correct number of labels for the augmented samples
y_holdout_noisy = np.repeat(y_holdout, np.ceil(num_augmented_samples / num_original_samples).astype(int))[:num_augmented_samples]

print(f"Noisy holdout set shape: {X_holdout_noisy.shape}")
print(f"Noisy holdout labels shape: {y_holdout_noisy.shape}")

# Load the best model
model = keras.models.load_model('best_model.h5')

# Evaluate on the noisy holdout set
print("\nEvaluating on the Noisy Holdout Set...")
y_holdout_pred = model.predict(X_holdout_noisy)
y_holdout_pred_classes = np.argmax(y_holdout_pred, axis=1)

# Calculate accuracy for the noisy holdout set
holdout_accuracy = accuracy_score(y_holdout_noisy, y_holdout_pred_classes)
print(f"Noisy Holdout Set Accuracy: {holdout_accuracy:.4f}")

# Generate classification report for the noisy holdout set
print("\nClassification Report for Noisy Holdout Set:")
print(classification_report(y_holdout_noisy, y_holdout_pred_classes, digits=4))

# Generate confusion matrix for the noisy holdout set
conf_matrix_holdout = confusion_matrix(y_holdout_noisy, y_holdout_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_holdout, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Noisy Holdout Set')
plt.show()




