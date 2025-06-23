# CNN-Based Image Classification Project

## Overview
This project followed a structured machine learning pipeline for building and evaluating Convolutional Neural Network (CNN) models for both image classification and emotion recognition tasks. The general process adopted is as follows:

### Data Collection & Preprocessing
- **Datasets Used**:
  - Emotion Recognition dataset (Kaggle) – 7 emotional categories (e.g., happy, sad, fear).
  - Custom Image Classification dataset – images collected from different devices.
- **Challenges**: Variations in image quality, resolution, lighting, and noise due to heterogeneous sources.
- **Preprocessing Techniques**:
  - **Image Normalization** – Scaling pixel values to [0, 1].
  - **Data Augmentation** – Flips, rotations, zooms to increase diversity.
  - **Resizing & Cropping** – Standardizing image dimensions while preserving aspect ratio.
  - **One-Hot Encoding** – For categorical label transformation (used in Approach 2).
  - **ImageDataGenerator** – Used in Approach 1 for pipeline-based data feeding.

 ### Model Architecture & Training
 **Model 1: ImageDataGenerator-based CNN**
 - **Architecture**:
   - Multiple Conv2D + ReLU + MaxPooling layers
   - Dense layer with dropout
   - Output via softmax for multi-class classification
   - **Issues Observed**: Underfitting & High Training Time

**Model 2: One-Hot Encoded Input CNN**
- **Architecture**:
  - Smaller, efficient Conv2D stacks
  - ReLU + MaxPooling + Dropout
  - Final Dense output layer
  - **Result**: Improved accuracy and drastically reduced training time.

 ### Evaluation & Analysis
 - **Metrics Used**:
   - Accuracy (Training, Validation, Test)
   - Loss Curves
   - Confusion Matrices
     
### Key Findings:
- Model 2 outperformed Model 1 in both training speed and generalization.
- Accuracy was significantly better using One-Hot Encoding for both datasets.

## Environment & Tools

**Language:**  
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)

**Libraries Used:**  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?logo=tensorflow)  ![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras)  ![NumPy](https://img.shields.io/badge/NumPy-1.20+-013243?logo=numpy)  ![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?logo=opencv)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?logo=scikit-learn)

**Development Tool:**  
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-Enabled-F37626?logo=jupyter)

**Hardware:**  
![MacBook Pro](https://img.shields.io/badge/MacBook%20Pro-Apple%20M1%20Pro-lightgrey?logo=apple&logoColor=white)
