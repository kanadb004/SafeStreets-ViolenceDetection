# SafeStreets Readme

## Overview  
This repository contains a real-time video-based violence detection system using a CNN-LSTM model. The project enables users to upload video footage through a web interface and receive instant feedback on whether any violent activity is detected. It aims to enhance public safety and surveillance using deep learning.

## Table of Contents  
- Features  
- Technologies Used  
- Usage  

## Features  
Violence Detection: Utilizes a CNN-LSTM model to detect violent activity in surveillance videos.

Web Interface: Upload videos via a clean frontend built with TailwindCSS and receive real-time feedback.  

Model Testing: Includes a Python script to directly test the trained model on video files.  

Model Training: Contains model training code and preprocessing steps for building the violence detection system.  

## Technologies Used  
Python  

TensorFlow/Keras (for CNN-LSTM model)  

OpenCV (for video processing)  

Flask (for backend server)  

TailwindCSS/HTML (for frontend UI)  

Jupyter Notebook (for training/testing pipeline)  

## Usage  
- Violence Detection: Upload a video through the web interface. The backend processes it and returns whether violence is detected.  
- Model Testing: Use the `test_predict.py` script to directly run the model on a local video file and get results.  
- Model Training: Use the Jupyter notebook in the `model/` directory to train or retrain the CNN-LSTM model.  

Violence Detection: Upload a video through the web interface. The backend processes it and returns whether violence is detected.  
Model Testing: Use the `test_predict.py` script to directly run the model on a local video file and get results.  
Model Training: Use the Jupyter notebook in the `model/` directory to train or retrain the CNN-LSTM model. We were not able to integrate the ML model with frontend prediction display due to time constraint.
