# Yoga-Pose-Classifier


This project classifies yoga poses using a 1D Convolutional Neural Network (CNN) built with Keras and TensorFlow.

## Algorithm Used

- 1D Convolutional Neural Network (CNN)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Evaluation: Accuracy, Precision, Recall, F1 Score

## Files

- `preprocessing.ipynb` – Load and prepare the dataset.
- `training.ipynb` – Build, train, and save the model.
- `testing.ipynb` – Load the saved model and evaluate on test data.
- `model.h5` – Trained model file.
- `labels.npy` – Numpy file containing all class labels.

## Dataset

- Input: `data.csv` with feature columns and a `Pose_Class` label column.
- Output: Trained model and metrics.

## How to Use

1. Run `preprocessing.ipynb` to prepare data.
2. Run `training.ipynb` to train the model and save it as `model.h5`.
3. Run `testing.ipynb` to evaluate the model performance.

## Requirements

Install the required packages:

pip install tensorflow keras pandas numpy scikit-learn matplotlib
