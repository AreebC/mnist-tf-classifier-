# MNIST Digit Classifier with TensorFlow

This project trains a neural network to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.

# Project Overview
- Trains the neural network on the MNIST image dataset
- A fully connected dense network is used as a baseline
- Improves performance with a CNN
- Used Docker and virtual environments to create a clean structure

# Improvements
- Baseline model does not capture spatial features of images as effectively as CNN model
- Convolutional layers capture edges and textures
- Pooling layers reduce complexity and keeps important features
- These layers lead to a better performance 

# Model performance
- Dense network - accuracy 97.75%
- CNN model - accuracy 98.85%

# How to run
- Option 1: Terminal
- git clone https://github.com/AreebC/mnist-tf-classifier-.git
- cd mnist-tf-classifier
-Optional: Create a virtual environment
  - python3 -m venv .venv
  - source .venv/bin/activate    # On Windows: source .venv/scripts/activate
- pip install -r requirements.txt
- python3 src/train.py
- Option 2: Docker - Needs Docker to be installed
- docker build -t mnist-cnn .
- docker run mnist-cnn
  
