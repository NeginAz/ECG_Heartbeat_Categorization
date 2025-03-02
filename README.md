# ECG Heartbeat Categorization Project

## Introduction

This project involves building a machine learning model to classify ECG signals into distinct heartbeat categories using the ECG Heartbeat Categorization Dataset. The objective is to create a robust classification model, enhance data diversity through augmentation, and develop a deployment-ready solution with a focus on model performance and real-world applicability.

## Dataset Overview

The dataset used for this project is sourced from Physionet's MIT-BIH Arrhythmia Dataset. It consists of 109,446 samples categorized into five classes of heartbeat types: Normal (0), Supraventricular ectopic (1), Ventricular ectopic (2), Fusion (3), and Unknown (4). The ECG signals are preprocessed, segmented, and sampled at 125 Hz, with each segment representing a single heartbeat with 187 data points.

## Project Objectives

1. **Data Processing:** Perform EDA, apply data augmentation, and handle missing data.
2. **Model Training:** Develop a CNN-LSTM model, handle class imbalance, and optimize performance.
3. **Testing on Holdout Set:** Simulate data shifts and evaluate performance.
4. **Deployment:** Implement a Flask API for model inference and discuss deployment strategies.


## Project Structure

The repository is organized into the following directories:

### src/ - Core Model & Augmentation Implementations
This folder contains Python classes and functions used across different notebooks. It includes:

cnn_lstm_classifier.py – Implementation of a CNN-LSTM-based classifier.
generate_synthetic_data_gmm.py – GMM-based data augmentation.
gan_time_series_generator.py – Generative Adversarial Network (GAN) for synthetic ECG data.
data_augmentation.py – Functions for augmentation techniques like noise addition and time shifting.
dimensionality_reduction.py – PCA and t-SNE implementations for feature visualization.
class_data_resampler.py – Resampling techniques for handling imbalanced datasets.
utils.py – Utility functions, including stratified sampling.
 A [README](src/SRC_README.md) inside this folder explains the contents and usage of each script.

### notebooks/ - Exploratory & Experimental Notebooks
This directory contains Jupyter notebooks implementing different approaches for ECG classification:

EDA.ipynb – Exploratory Data Analysis (EDA) and feature visualization.
GMM.ipynb – Data augmentation using Gaussian Mixture Models with classification.
GAN.ipynb – Synthetic data generation with a GAN and classification.
Resample.ipynb – Resampling-based data augmentation with classification.
No_Augmentation.ipynb – Baseline classification model without augmentation.
[README](notebooks/NOTEBOOKS_README.md) has detailed explanations of the methodology and findings.

### data/ - Dataset, Models & Figures
CSV files: Contains the original MIT-BIH ECG dataset used for training and testing.
Saved models: Pre-trained model weights for evaluation.
Figures: Plots and visualizations generated during analysis.


### api/ - Flask Application for Model Deployment
This folder contains a minimal Flask app to serve the trained ECG classification model:

app.py – The main API backend for handling requests.
index.html – A simple frontend for user interaction.
📌 See the [README.md](api/API_README.md) in this folder for deployment instructions.


## Data Processing
- **EDA:** Visualized class distribution and identified data imbalance.
- **Data Augmentation:** Implemented with Gaussian Mixture Model (GMM) and custom augmentation class (`DataAugmentation`).
- **Feature Engineering:** Added statistical features such as mean, standard deviation, skewness, and kurtosis.
- **Handling Imbalance:** Synthetic samples generated using GMM for minority classes.

### 1. Exploratory Data Analysis (EDA):
Before training any model, we conducted an in-depth analysis of the dataset:

- Dataset Structure: Checked the dimensions of both the training and test sets.
- Sample Inspection: Observed a few samples of ECG signals to understand their nature.
- Class Distribution: Visualized the distribution of heartbeat categories to assess dataset balance.
- Feature Statistics: Computed statistical properties of the train and test sets.

###  2.Data Preparation
 - Dataset Splitting:
    - Created a validation set from the training set to ensure fair model evaluation.
    Divided the test set into:
    - Regular Test Set (used for final evaluation).
    - Holdout Set (later used to test model robustness against noisy and shifted data).
    The validation set was balanced, meaning each class had an equal number of samples. 

### 3. Data Augmentation Techniques
 
To handle the class imbalance issue, we experimented with three augmentation strategies:

#### 3.1 Gaussian Mixture Model (GMM)

We trained Gaussian Mixture Models (GMM) on the minority classes.
The trained GMMs generated synthetic ECG signals to augment the dataset.


#### 3.2 Generative Adversarial Networks (GANs)
-  Implemented multiple versions of GAN-based signal generation, inspired by prior -  research on ECG synthesis.
- The generator models were designed using LSTM and CNN layers to capture temporal dependencies.
- The discriminator networks used 1D Convolutional layers to distinguish real and fake signals.

#### 3.3 Resampling (Upsampling)
Used bootstrapping (resampling with replacement) to duplicate underrepresented class samples.
The goal was to increase representation without introducing synthetic data.

### 4. Additional Pre-Training Augmentation
Before training, additional data augmentation techniques were applied directly to the original ECG signals to improve model generalization:

#### 4.1. Time Shifting
Each ECG signal was shifted slightly forward or backward in time.
Helps the model become invariant to phase shifts, making it more robust.
#### 4.2. Noise Addition

Random Gaussian noise was added to the signals to simulate real-world variations.
Prevents overfitting and forces the model to learn more generalizable features.

### 5. Model Selection and Training
Implemented a CNN-LSTM hybrid model for ECG classification.

### 6. Model Evaluation
Evaluated the model on the validation set and test set.
Used metrics including accuracy, precision, recall, F1-score, and confusion matrix.
Compared model performance across datasets with and without augmentation.

### 7. Robustness Testing on Holdout Set
Introduced noise and time shifts to test model stability.
Assessed how well the model generalized under data distribution shifts.

### 8. Model Deployment Strategy
Designed an inference pipeline to make real-time predictions on ECG data.
Discussed deployment considerations including scalability, versioning, and monitoring.

- **API:** Developed a Flask API (`app.py`) to accept CSV uploads and predict heartbeat classes.
- **Frontend:** Simple web interface for file upload and result display.
- **Model Inference:** The API loads the trained model (`saved_model.h5`) and processes input data.

## Project Structure Tree

```
ECG_Heartbeat_Categorization/
│
├── api/                    # API for ECG data processing
│   ├── app.py              # Flask application
│   ├── generate_samples.py # Script to generate synthetic samples
│   ├── API_README.md       # Documentation for API usage
│   ├── templates/          # HTML templates for API
│   │   └── index.html
│   ├── synthetic_ecg_data_no_headers.csv # Example synthetic ECG data
│
├── data/                   # Raw and processed data files
│   ├── mitbih_train.csv    # Training dataset
│   ├── mitbih_test.csv     # Test dataset
│   ├── saved_model/        # Folder containing trained models
│   │   ├── GMM/saved_model.h5
│   │   ├── Resample/saved_model.h5
│   │   ├── no_augmentation/saved_model.h5
│
├── figures/                # Saved plots and figures for analysis
│   ├── EDA/                # Exploratory Data Analysis figures
│   │   ├── Autocorrelation_Class_0.0.png
│   │   ├── Autocorrelation_Class_1.0.png
│   │   ├── Frequency_Domain_Class_2.0.png
│   │   ├── ECG_Signal_Waveforms_for_Class_4.0.png
│   ├── Model Accuracy.png  # Model performance visualization
│   ├── Confusion Matrix on Test Set.png
│   ├── Distribution_of_Class_Labels.png
│
├── notebooks/              # Jupyter notebooks for analysis & model training
│   ├── EDA.ipynb           # Exploratory Data Analysis notebook
│   ├── GMM.ipynb           # Gaussian Mixture Model-based generation
│   ├── Resample.ipynb      # Data resampling notebook
│   ├── No_augmentation.ipynb # Training without augmentation
│   ├── NOTEBOOKS_README.md # Documentation for notebooks
│
├── src/                    # Source code for model training & preprocessing
│   ├── cnn_lstm_classifier.py  # CNN-LSTM model for ECG classification
│   ├── data_augmentation.py    # Data augmentation techniques
│   ├── generate_synthetic_data_gmm.py  # GMM-based synthetic ECG generation
│   ├── class_data_resampler.py  # Resampling ECG data for balance
│   ├── dimensionality_reduction.py  # Dimensionality reduction utilities
│   ├── gan_time_series_generator.py  # GAN-based time-series generation
│   ├── utils.py               # Helper functions
│   ├── SRC_README.md          # Documentation for source files
│
├── requirement.txt           # Python dependencies (ensure correct name)
├── README.md                 # Main project documentation
├── .gitignore                # Git ignore file (for excluding unnecessary files)
```

## Installation

1. Clone the repository:

```sh
git clone https://github.com/yourusername/ECG_Heartbeat_Categorization.git
cd ECG_Heartbeat_Categorization
```

2. Create a virtual environment and install dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Key Takeaways

Data augmentation significantly improved classification performance, especially for underrepresented classes.
GANs were the most effective method for generating realistic ECG waveforms, but GMM-based augmentation was computationally cheaper.
The model was sensitive to data shifts and noise, requiring additional robustness testing.

## Future Work

- Enhance feature engineering with domain-specific ECG metrics.
- Explore advanced augmentation techniques.
- Consider model ensemble strategies to improve minority class performance.

## Acknowledgments

This project was developed as part of a technical test for a Data Science role. Special thanks to Physionet for the dataset and the reviewers for the opportunity to demonstrate my technical skills.

