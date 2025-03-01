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
- Visualization:
  - Plotted a few raw ECG signals.
  - Used bar plots to show class distributions.
 - Applied Principal Component Analysis (PCA) and T-SNE to project high-dimensional data into two components for better insight.
 - As ECG signals are sequential, each data point depends on the preceding and succeeding points. 
<p align="center">
<img src="https://github.com/user-attachments/assets/bfedce1b-30c8-4789-980b-29073e376767" alt="Sample Image" width="500">
</p>  

<div align="center">
    <img src="https://github.com/user-attachments/assets/d3b6d6dc-d572-49dc-b710-49008ef77238" width="500">
    <img src="https://github.com/user-attachments/assets/2e045c68-945a-4592-8139-af8f8a7848bd" width="500">
    <img src="https://github.com/user-attachments/assets/909ce610-830b-436b-9d1d-8a5f3401f94e" width="500">
    <img src="https://github.com/user-attachments/assets/4152fb0d-f34a-483e-a194-1d2b911fbed6" width="500">
    <img src="https://github.com/user-attachments/assets/941965e3-df70-403b-bfa6-a3c04ff2dd8a" width="500">
</div>


Findings: 
Classes 1, 2, and 3 were significantly underrepresented in the dataset, highlighting the need for data augmentation.

###  2.Data Preparation
 - Dataset Splitting:
    - Created a validation set from the training set to ensure fair model evaluation.
    Divided the test set into:
    - Regular Test Set (used for final evaluation).
    - Holdout Set (later used to test model robustness against noisy and shifted data).
    The validation set was balanced, meaning each class had an equal number of samples. (how many ??? )

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
Applied batch normalization, dropout regularization, and early stopping to prevent overfitting.
Hyperparameter tuning was conducted to optimize model performance.

<p align="center">
<img src="https://github.com/user-attachments/assets/9213b55d-417a-4a1e-8d30-7f7a89ffbca5" alt="Sample Image" width="900">
</p>  

The model is a hybrid CNN-LSTM architecture designed for time-series classification, as follows: 
#### Convolutional Layers (Conv1D):
- The first two Conv1D layers extract local patterns from the ECG signals.
- The kernel size 3 allows the model to detect small changes in waveform.
#### Batch Normalization: 
- Stabilizes training and speeds up convergence

#### MaxPooling1D: 
- Reduces dimensionality while preserving the most important features.
  
#### Dropout Layers:
- Prevents overfitting by randomly deactivating some neurons during training.
  
#### LSTM Layer:
- Captures long-range dependencies in the ECG signal (detecting irregularities)
  
#### Dense Layers with Regularization:
- The dense layer adds more learning capacity.
- L2 regularization prevents overfitting. 

#### Softmax Output Layer:
- Used for multi-class classification (each ECG signal is assigned a probability for each class).

This model is effective for ECG classification because CNNs extract spatial features, and LSTMs learn temporal relationships, making it suitable for time-series data like ECG signals.

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



## Results

- Validation Accuracy: \~92%
- Test Accuracy: \~84%
- Holdout Set Performance: Analyzed and discussed the impact of data shifts.

For the first model, the LSTM-CNN without any data augmentation, the figures are below:

<p align="center">
<img src="https://github.com/user-attachments/assets/1a099712-f1f5-4889-b35e-78bd46dd4b65" alt="Sample Image" width="300">
</p>  


<div align="center">
    <img src="https://github.com/user-attachments/assets/64842f70-0c39-4dbd-88ad-185616c12c10" width="250">
    <img src="https://github.com/user-attachments/assets/9ee97c5e-70ee-48ce-b6be-59a8a645d29f" width="250">
</div>


## Project Structure

```
ECG_Heartbeat_Categorization/
│
├── data/                   # Raw and processed data files
├── notebooks/              # Jupyter notebooks for EDA, model training, and testing
├── src/                    # Source code including model, augmentation, and API
│   ├── cnn_lstm_classifier.py
│   ├── data_augmentation.py
│   └── app.py              # Flask application
├── templates/              # HTML templates for the web interface
│   └── index.html
├── requirements.txt        # Python package requirements
└── README.md               # Project documentation
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


### Run the Flask App

```sh
cd src
python app.py
```

Navigate to `http://localhost:5000` in your browser.


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

