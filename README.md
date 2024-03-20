# Sign-Language-Detection-using-ACTION-RECOGNITION

This repository contains the code and documentation for a sign language detection system using action recognition techniques. The project utilizes mediapipe holistics for collecting keypoints and trains a Long Short-Term Memory (LSTM) Deep Learning model to recognize sign language from video input.

# Overview

The project is divided into two main Jupyter notebooks:

SignLanguageDetection.ipynb: This notebook covers the entire process from data collection and preprocessing to model training and evaluation.
SignLanguageDetectionRealTimePrediction.ipynb: This notebook demonstrates how to use the trained model for real-time sign language prediction.
Installation
Before running the notebooks, ensure you have all the necessary dependencies installed. Both notebooks require similar dependencies, primarily focusing on mediapipe for keypoints extraction and TensorFlow for model training and prediction.

## Installation

To install the required packages, use the following pip commands:

- Python Version 3.10.10
- `!pip install tensorflow==2.13.0`
- `!pip install mediapipe==0.10.3`
- `!pip install scikit-learn==1.3.1`
- `!pip install opencv-python==4.9.0.80`
- `!pip install matplotlib==3.8.3`


# Notebooks Content

# SignLanguageDetection.ipynb

Installing and importing dependencies: Instructions on setting up your environment with the necessary libraries.
Collecting keypoints from mediapipe holistics: Demonstrates how to use mediapipe to collect keypoints from video data.
Extracting keypoints value: Code for extracting keypoints values for further processing.
Folder setup for data Collection: Guidelines on organizing your data for training and testing.
COLLECTING KEYPOINTS FOR TRAINING AND TESTING: Steps for collecting keypoints data that will be used for training the LSTM model.
Preprocessing our data: Preprocesses the collected data to make it suitable for training.
Building and training LSTM DL Model: Details the construction and training of the LSTM model using TensorFlow.
Predicting result from test data: Demonstrates how to use the trained model to make predictions on new data.
Saving weights for future: Instructions on saving the trained model weights for future use.
Accuracy Calculation and Evaluation using Confusion matrix: How to evaluate the model's performance using a confusion matrix.

# SignLanguageDetectionRealTimePrediction.ipynb

Focuses on using the trained model from SignLanguageDetection.ipynb for real-time prediction. It includes steps for importing dependencies, setting up mediapipe for keypoints extraction, and predicting sign language from live video input.

# Usage

Start by running the SignLanguageDetection.ipynb notebook to train your model. Follow the instructions within the notebook for data collection, training, and evaluation.
Once the model is trained and saved, use the SignLanguageDetectionRealTimePrediction.ipynb notebook for real-time sign language detection. Ensure your environment is correctly set up with the required libraries and the trained model weights are accessible.

# Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes or enhancements.


To acknowledge the reference or inspiration for your project, you can add a section to your README.md titled "Acknowledgments". This section can be used to thank individuals, organizations, or even repositories that have significantly contributed to the development of your project. Here's how you could structure this addition:

# Acknowledgments

This project on Sign Language Detection using Action Recognition was inspired by and developed with reference to various resources and tutorials provided by Nick Nochnack. His contributions to the field of computer vision and machine learning, particularly his work on action recognition and sign language detection, have been invaluable to the development of this project.

We would like to extend our sincerest gratitude to Nick Nochnack for making his resources available to the public, which have greatly facilitated our learning and development process. His GitHub repository, which contains a wealth of knowledge and code examples, can be found here: Nick Nochnack's GitHub - https://github.com/nicknochnack

A special thanks goes out to everyone who has contributed to the aforementioned repository, as their work has directly influenced the successful completion of our project.

