# Yoga-Pose-Classifier

This project is a deep learning-based real-time yoga pose classification system using MediaPipe and Keras. It enables users to perform yoga poses in front of a webcam, classifies them accurately, and provides feedback on whether the pose is correct. Additionally, it logs the duration and time of each pose, helping track progress over time.

## Algorithm Used

- 1D Convolutional Neural Network (CNN)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Evaluation: Accuracy, Precision, Recall, F1 Score

## Files

- `preprocessing.ipynb` – Preprocesses pose images using MediaPipe to extract and normalize 3D body landmarks, saving them with corresponding class labels.
- `training.ipynb` – Build, train, and save the model.
- `testing.ipynb` – Performs real-time yoga pose classification using a trained model and logs the duration and timestamp of correct poses into a CSV file
- `model.h5` – Trained model file.
- `labels.npy` – Numpy file containing all class labels.

## Dataset

- Input: Takes in images extracts angles of the pose save the angles in data.csv
- Output: Trained model and metrics.

## How to Use

1. Run `preprocessing.ipynb` to prepare data.
2. Run `training.ipynb` to train the model and save it as `model.h5`.
3. Run `testing.ipynb` to evaluate the model performance.

## Requirements

Install the required packages:

pip install tensorflow keras pandas numpy scikit-learn matplotlib
