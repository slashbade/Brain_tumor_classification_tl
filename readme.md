# Brain Tumor Classification

Final Project for Principals of Machine Learning

## Abstract of Work

This project works on Brain Tumor MRI image classification dataset, and is divided into 2 independent tasks, which is **image compression** task and **tumor type classification** task. For the first one, it can be viewed as an unsupervised learning task, and is solved by **Principal Component Analysis (PCA)** and **Convolutional Autoencoder** method. 

For the second one, it is treated as a supervised classification task, and is tackled with **Support Vector Classifier (SVC)**, **Logisitic Regression**, **Convolutional Neural Network (CNN)**, and **Transfer Learning method** featuring MobileNet V2 as a base model. In addition, the project adopts **cross validation procedure** in the model selection period for classical machine learning methods. And a brief analysis with visualization of dataset is placed ahead of application of methods.

In terms of the results, the projects finds convolutional autoencoder have the best reconstruction performance, with MSE loss below 0.01 in both train and test set. And Transfer Learning adapted from MobileNet V2 have the best classification performance, with F1-score over 0.90 on train set and validation set, over 0.80 on given real-world test set. Furthermore, some interesting findings of results for misclassified samples is given and explained for several methods.

**Note** This project puts some utility functions into utils.py, please make sure to run in the same directory. 

## Introduction to Dataset

A Brain tumor is considered as one of the most aggressive diseases, among children and adults. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately 34 percent for men and 36 percent for women. And the best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). The dataset is contributed by Navoneel Chakrabarty and Swati Kanchan.

This dataset provides a huge amount of image data generated through the scans. These images are typically examined by the radiologist. However, a manual examination can be error-prone due to the complexities. Hence, proposing a system performing detection and classification is the main goal of this dataset.

## Usage

