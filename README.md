## Wallet Anomaly Detection

This repository contains code for detecting anomalies in user balance claims using various machine learning techniques including Isolation Forest, Autoencoders, and DBSCAN.

## Table of Contents
Project Overview
Dataset
Installation
Usage
Methodology
Isolation Forest
Autoencoder
DBSCAN
Results
Contributing
License

## Project Overview

The goal of this project is to identify anomalies in user balance claims by applying multiple machine learning techniques. Anomalies are detected using Isolation Forest, Autoencoder, and DBSCAN models, and the results are combined to identify potential outliers.

## Dataset

The dataset used in this project is an Excel file named User Ballance Claim Approved_1.xlsx containing user balance claims. The dataset is converted to CSV format for easier processing.

## Installation
To get started with this project, clone the repository and install the necessary dependencies:

bash
Copy code
git clone https://github.com/yourusername/wallet-anomaly-detection.git
cd wallet-anomaly-detection
pip install -r requirements.txt


The requirements.txt file should include the following libraries:

Copy code
pandas
numpy
seaborn
matplotlib
scikit-learn
keras

## Usage
## Preprocessing:
Convert the Excel file to CSV format.
Parse datetime fields and select numerical and boolean columns for analysis.

## Isolation Forest:
Train the Isolation Forest model on the dataset.
Identify anomalies based on the contamination parameter.

## Autoencoder:
Normalize the data using StandardScaler.
Reduce dimensionality using PCA.
Train an Autoencoder model to identify reconstruction errors.

## DBSCAN:
Apply DBSCAN in chunks to handle large datasets.
Identify anomalies based on clustering.
Combine Results:
Combine the results from Isolation Forest, Autoencoder, and DBSCAN.
Save the final anomalies to an Excel file.
Methodology
Isolation Forest
Isolation Forest is an unsupervised learning algorithm for anomaly detection that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

python
Copy code
from sklearn.ensemble import IsolationForest

isolation_forest_model = IsolationForest(contamination=0.01)
isolation_forest_model.fit(X)
predictions = isolation_forest_model.predict(X)
Autoencoder
An Autoencoder is a type of neural network used to learn efficient codings of unlabeled data. It consists of an encoder and a decoder. The model is trained to minimize the reconstruction error.

python
Copy code
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=3, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='linear'))
model.compile(optimizer='adam', loss='mse')
model.fit(X_pca, X_pca, epochs=50, batch_size=32, validation_split=0.1)
DBSCAN
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are closely packed together while marking points that are in low-density regions as outliers.

python
Copy code
from sklearn.cluster import DBSCAN

dbscan_model = DBSCAN(eps=3, min_samples=5)
chunk_outliers = dbscan_model.fit_predict(X_scaled[start_idx:end_idx]) == -1
Results
The final results are saved in an Excel file Partner_results_anomalies_withdrawl.xlsx containing the identified anomalies.

python
Copy code
partner_results.to_excel('D:\\wallet anomaly\\Partner_results_anomalies_withdrawl.xlsx')
