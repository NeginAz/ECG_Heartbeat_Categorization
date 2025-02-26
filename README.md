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

## Data Processing

- **EDA:** Visualized class distribution and identified data imbalance.
- **Data Augmentation:** Implemented with Gaussian Mixture Model (GMM) and custom augmentation class (`DataAugmentation`).
- **Feature Engineering:** Added statistical features such as mean, standard deviation, skewness, and kurtosis.
- **Handling Imbalance:** Synthetic samples generated using GMM for minority classes.

## Model Training

- **Model Architecture:** A CNN-LSTM hybrid model with early stopping and learning rate scheduling.
- **Imbalance Handling:** Applied oversampling and augmentation strategies.
- **Validation Strategy:** 5-fold cross-validation with StratifiedKFold.
- **Evaluation Metrics:** Accuracy, recall, precision, F1 score, and confusion matrix.

## Holdout Set Testing

- Created a holdout set to simulate real-world data shifts.
- Introduced noise using the `DataAugmentation` class.
- Evaluated performance degradation and analyzed results.

## Deployment

- **API:** Developed a Flask API (`app.py`) to accept CSV uploads and predict heartbeat classes.
- **Frontend:** Simple web interface for file upload and result display.
- **Model Inference:** The API loads the trained model (`best_model.h5`) and processes input data.

### Run the Flask App

```sh
cd src
python app.py
```

Navigate to `http://localhost:5000` in your browser.

## Results

- Validation Accuracy: \~92%
- Test Accuracy: \~84%
- Holdout Set Performance: Analyzed and discussed the impact of data shifts.

## Future Work

- Enhance feature engineering with domain-specific ECG metrics.
- Explore advanced augmentation techniques.
- Consider model ensemble strategies to improve minority class performance.

## Acknowledgments

This project was developed as part of a technical test for a Data Science role. Special thanks to Physionet for the dataset and the reviewers for the opportunity to demonstrate my technical skills.

