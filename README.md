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

In the begining, I started by exploring the dataset. First, checking the dimensions of the train and test sets, and number of samples in both sets. In order to learn more about the nature of the datset, I started by obserivng a few samples of the dataset, to know the features. Later, I checked the distribution of labesl, to see how balanced the dataset is. 
We observed that class number 3, 1 and 2 are significantly undersampled. Thus, this dataset needs data augmentation techniques to even the data for training. 

Then, I startd by computing the statistical measures of the train and test set. Then plotting a few samples, and bar plot of class distributions. I then plotted the first two components of train set using PCA and T-SNE. to see the clusters and give more insight for furthur analysis. Then, I sepetated the validation set from the train set. The augmentation techniques will be applied only on the traning set, and the validation set is remained the same. The test set is also broken down into half, a test set and a hold-out set. The hold-out set will be later be modfied to test the model against noisy data and time shifts, to measure the robustness of the model. For the validaion  set, the equal number of samples from each class was selectd to make sure the the training is tested on a fair basis. 
Later I explored three differnet methods for data generation: using gussuan mixture models, GAN, and resampling (upsamppling the signal). 

1 -Gussian Mixture Models: For the three classes with the least nunber of samples, we used this method to genrate more samples. By plotting the T-SNE and PCA, to find out how many components are needded. Based on the plots, one and two components were generated. The samples generated were noisy and not similar to the original ones. 

2- The second method was using GANs to generate more samples for the undersampeld classes. I implemneted and tested different GANs. 

3- 
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

