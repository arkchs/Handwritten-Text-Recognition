# Handwritten Text Recognition with K-Nearest Neighbors (KNN)
This repository contains code and instructions for building a handwritten text recognition system using the K-Nearest Neighbors (KNN) algorithm. We’ll leverage the MNIST dataset, which consists of grayscale images containing handwritten digits (0-9).

## Prerequisites

Python 3.12.3 <br>
NumPy 1.26.4 <br>
Scikit-learn  1.5.0 <br>
Jupyter 1.0.0 <br>

## Getting Started
Clone this repository:<br>
`git clone https://github.com/your-username/handwritten-text-recognition.git` <br>
`cd handwritten-text-recognition`

## Download the MNIST dataset:
You can download the dataset from the official MNIST website or use the following command: <br>
`wget https://github.com/mniist/mniist/raw/main/mnist.pkl.gz`

## Implement KNN:
Split the data into training and testing sets.
Train the KNN model on the training data.
Evaluate the model on the testing data.

## Run the code:
`python knn_handwritten_recognition.py`

## Results
Our KNN model achieved an accuracy of approximately 96% on the MNIST test set. You can further fine-tune hyperparameters or explore other algorithms to improve performance.

## Acknowledgments
The MNIST dataset is available at http://yann.lecun.com/exdb/mnist/.
Inspired by Scikit-learn’s KNN documentation.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
