# README for Deep Learning Model with TensorFlow

## Overview
This repository contains the Python code for a deep learning model using TensorFlow to predict employee attrition. The project is structured to include separate modules for data loading, preprocessing, model building, and evaluation.

## Project Structure

- `main.py`: The main script that orchestrates the data loading, preprocessing, model training, evaluation, and plotting of training curves.
- `data_preprocessing/`: Directory containing modules for loading and preprocessing data:
  - `load_data.py`: Contains the function `load_data` to load data from a specified CSV file.
  - `preprocess_data.py`: Contains the function `preprocess_data` to preprocess the loaded data.
- `models/`: Directory containing the model architecture:
  - `build_model.py`: Contains the function `build_model` to create a TensorFlow model based on the input shape.
- `evaluation/`: Directory containing the evaluation logic:
  - `evaluate_model.py`: Contains the function `evaluate_model` to evaluate the model's performance using test data.

## Setup Instructions

### Prerequisites
- Python 3.8 or above
- pip (Python package installer)

### Dependencies
To install the required packages, run the following command in your terminal:
```bash
pip install tensorflow pandas matplotlib
```

### Running the Code
To run the main script, navigate to the project directory in your terminal, and execute:
```bash
python main.py
```

## Functionality

1. **Data Loading**: The script starts by loading the data from `utils/employee_attrition.csv` using the `load_data` function.
2. **Data Preprocessing**: The loaded data is then preprocessed to be suitable for training using the `preprocess_data` function. This typically involves normalizing features, encoding categorical variables, and splitting the data into training and testing sets.
3. **Model Building**: A deep learning model is constructed using the `build_model` function. The architecture depends on the input shape derived from the training data's features.
4. **Model Training**: The model is trained over 50 epochs with the training data. The training progress is silent (`verbose=0`).
5. **Model Evaluation**: After training, the model predictions are rounded and compared to the actual test labels using the `evaluate_model` method.
6. **Plotting**: The training loss curves are plotted to help visualize the model's learning progress over epochs.

## Visualization
After running the script, a plot will be displayed showing the training and validation loss over the epochs, aiding in assessing the model's performance and convergence.

## Extending the Code
To adapt or extend the code:
- Modify `load_data.py` and `preprocess_data.py` for different datasets or preprocessing techniques.
- Enhance `build_model.py` to experiment with different model architectures or to fine-tune hyperparameters.
- Update `evaluate_model.py` for more comprehensive metrics or custom evaluation strategies.

