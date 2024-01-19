# Patrick Ryan Project 4

## Overview
This Jupyter Notebook is a comprehensive machine learning script designed for a complete machine learning workflow. It includes data loading, preprocessing, exploratory analysis, model building (including deep learning models), hyperparameter tuning, and model evaluation.

## Requirements
To run this notebook, the following Python libraries are required:
- Data Manipulation and Analysis: `pandas`, `numpy`
- Logging and Mathematical Operations: `logging`, `math`
- Data Visualization: `matplotlib.pyplot`, `seaborn`
- Machine Learning - Preprocessing and Model Selection: Libraries from `sklearn` like `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `train_test_split`, `KFold`
- Machine Learning - Metrics: Metrics such as `classification_report`, `accuracy_score`
- Machine Learning - Algorithms: `RandomForestClassifier`, `SelectFromModel`
- Deep Learning: `tensorflow`, with specific imports for neural network layers, optimizers, regularizers, and callbacks
- Hyperparameter Tuning: `keras_tuner.tuners.Hyperband`, `keras_tuner.engine.hyperparameters.HyperParameters`

## Functionality
The script includes several key functionalities organized into function definitions for a structured machine learning approach:
1. **Data Loading and Preprocessing**: Functions like `load_data`, `normalize_and_encode`, `handle_missing_values`, `add_missing_columns` for preparing the dataset.
2. **Exploratory Data Analysis**: A function `exploratory_data_analysis` for data analysis and visualization.
3. **Feature Engineering**: Functions like `identify_data_types` and `feature_selection` for processing and selecting important features.
4. **Model Building and Training**: Includes deep learning models (`create_lstm_cnn_model`) and functions like `train_model`, `model_builder` for constructing and training models.
5. **Hyperparameter Tuning and Model Evaluation**: Functions such as `tune_hyperparameters`, `evaluate_model`, `cross_validate_model` for optimizing and assessing model performance.
6. **Utility and Helper Functions**: Functions like `prepare_sequence_data`, `lr_schedule` for specific tasks and adjustments during model training.

## Usage
To use this notebook:
1. Ensure all the required libraries are installed.
2. Run each cell in order, starting from the top.
3. Modify parameters, data inputs, or functions as needed for your specific use case.

## Main Execution Block
The script includes a main execution block, indicating it is designed to be run as a standalone program.

## Additional Notes
- The script does not contain class definitions or unique patterns, focusing instead on functional programming.
- It is well-structured for handling various aspects of a machine learning workflow, from data preparation to model training and evaluation.
 
