# Importing necessary libraries for data manipulation and analysis
import pandas as pd  # Pandas for handling data in tabular form
import numpy as np  # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for data visualization
import seaborn as sns  # Seaborn for statistical data visualization

# Importing modules from scikit-learn for machine learning tasks
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,  # Metrics for model evaluation
    ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve  # Confusion matrix and ROC curve
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder  # For data preprocessing
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.model_selection import GridSearchCV, cross_val_score  # Grid search and cross-validation
from sklearn.feature_selection import RFE  # Recursive Feature Elimination
from imblearn.over_sampling import SMOTE  # Handling class imbalance using Synthetic Minority Over-sampling Technique
import xgboost as xgb  # XGBoost library for gradient boosting
from xgboost import XGBClassifier  # XGBoost classifier

# Importing modules for logging and parallel processing
import logging  # Logging for tracking and debugging
from joblib import parallel_backend  # Joblib for parallel processing

# Set up logging
logging.basicConfig(level=logging.INFO)

def extract_percentile(value):
    """
    Extracts the percentile value from the 'AGE_PERCENTIL' column.

    Args:
        value (str): The value from the 'AGE_PERCENTIL' column.

    Returns:
        int: The extracted percentile value.
    """
    try:
        if 'Above' in value:
            return 90
        return int(value.replace('th', ''))
    except (ValueError, TypeError):
        return 0

def define_age_group(age):
    """
    Defines the age group based on the provided age.

    Args:
        age (float): The age value.

    Returns:
        str: The corresponding age group ('child', 'young_adult', 'middle_adult', 'older_adult', 'elderly').
    """
    if age <= 18:
        return 'child'
    elif 18 < age <= 35:
        return 'young_adult'
    elif 35 < age <= 50:
        return 'middle_adult'
    elif 50 < age <= 65:
        return 'older_adult'
    else:
        return 'elderly'

    if age <= 18:
        return 'child'
    elif 18 < age <= 35:
        return 'young_adult'
    elif 35 < age <= 50:
        return 'middle_adult'
    elif 50 < age <= 65:
        return 'older_adult'
    else:
        return 'elderly'

def handle_missing_values(df, numerical_cols, important_features=None):
    """
    Handles missing values in the DataFrame by either removing rows with missing values
    or imputing missing values based on the specified strategy.

    Args:
        df (pd.DataFrame): The input DataFrame containing missing values.
        numerical_cols (list): List of numerical columns for imputation.
        important_features (list, optional): List of important features. If provided, rows with
            missing values for these features will be removed.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    logging.info("Handling missing values...")
    original_rows = df.shape[0]  # Total number of rows in the original DataFrame
    
    # Check if any important features have missing values
    if important_features:
        logging.info("Removing rows with missing values for important features...")
        missing_cols = [col for col in important_features if col in df.columns and df[col].isnull().any()]
        if missing_cols:
            # Remove rows with missing values for important features
            df_cleaned = df.dropna(subset=missing_cols)
            remaining_rows = df_cleaned.shape[0]  # Number of rows after cleaning
            removed_rows = original_rows - remaining_rows
            logging.info(f"Removed {removed_rows} rows ({(removed_rows / original_rows) * 100:.2f}%) with missing values for important features.")
            logging.info(f"Percentage of data remaining: {(remaining_rows / original_rows) * 100:.2f}%")
        else:
            logging.info("No rows with missing values for important features. DataFrame remains unchanged.")
            df_cleaned = df.copy()
    else:
        logging.info("Removing all rows with missing values...")
        df_cleaned = df.dropna()
        remaining_rows = df_cleaned.shape[0]  # Number of rows after cleaning
        removed_rows = original_rows - remaining_rows
        logging.info(f"Removed {removed_rows} rows ({(removed_rows / original_rows) * 100:.2f}%) with missing values.")
        logging.info(f"Percentage of data remaining: {(remaining_rows / original_rows) * 100:.2f}%")

    # Convert numerical_cols to pandas Index
    numerical_cols = pd.Index(numerical_cols)

    # Impute missing values for numerical columns
    if not numerical_cols.empty:
        imputer_numeric = SimpleImputer(strategy='median')
        df_cleaned[numerical_cols] = imputer_numeric.fit_transform(df_cleaned[numerical_cols])

    # Impute missing values for categorical columns
    for col in df_cleaned.columns:
        if col not in numerical_cols:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    return df_cleaned

    logging.info("Handling missing values...")
    original_rows = df.shape[0]  # Total number of rows in the original DataFrame
    
    # Check if any important features have missing values
    if important_features:
        logging.info("Removing rows with missing values for important features...")
        missing_cols = [col for col in important_features if col in df.columns and df[col].isnull().any()]
        if missing_cols:
            # Remove rows with missing values for important features
            df_cleaned = df.dropna(subset=missing_cols)
            remaining_rows = df_cleaned.shape[0]  # Number of rows after cleaning
            removed_rows = original_rows - remaining_rows
            logging.info(f"Removed {removed_rows} rows ({(removed_rows / original_rows) * 100:.2f}%) with missing values for important features.")
            logging.info(f"Percentage of data remaining: {(remaining_rows / original_rows) * 100:.2f}%")
        else:
            logging.info("No rows with missing values for important features. DataFrame remains unchanged.")
            df_cleaned = df.copy()
    else:
        logging.info("Removing all rows with missing values...")
        df_cleaned = df.dropna()
        remaining_rows = df_cleaned.shape[0]  # Number of rows after cleaning
        removed_rows = original_rows - remaining_rows
        logging.info(f"Removed {removed_rows} rows ({(removed_rows / original_rows) * 100:.2f}%) with missing values.")
        logging.info(f"Percentage of data remaining: {(remaining_rows / original_rows) * 100:.2f}%")

    # Convert numerical_cols to pandas Index
    numerical_cols = pd.Index(numerical_cols)

    # Impute missing values for numerical columns
    if not numerical_cols.empty:
        imputer_numeric = SimpleImputer(strategy='median')
        df_cleaned[numerical_cols] = imputer_numeric.fit_transform(df_cleaned[numerical_cols])

    # Impute missing values for categorical columns
    for col in df_cleaned.columns:
        if col not in numerical_cols:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    return df_cleaned

def handle_class_imbalance(X_train, y_train):
    """
    Handles class imbalance in the target variable using Synthetic Minority Over-sampling Technique (SMOTE).

    Args:
        X_train (pd.DataFrame): The input features of the training set.
        y_train (pd.Series): The target variable of the training set.

    Returns:
        tuple: A tuple containing the resampled features (X_resampled) and the corresponding resampled target variable (y_resampled).
    """
    logging.info("Handling class imbalance...")
    print("Unique values in y_train before SMOTE:", np.unique(y_train))
    
    # Check if labels are numeric
    if np.issubdtype(y_train.dtype, np.number):
        # Convert to binary labels based on a threshold
        threshold = 0.5
        y_train_binary = (y_train > threshold).astype(int)
        # Use SMOTE to handle class imbalance
        smote = SMOTE(sampling_strategy='auto', random_state=42)  # 'auto' adjusts the strategy based on the input data
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train_binary)
        return X_resampled, y_resampled
    else:
        # If labels are not numeric, proceed with SMOTE without conversion
        smote = SMOTE(sampling_strategy='auto', random_state=42)  # 'auto' adjusts the strategy based on the input data
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

def feature_engineering(df_encoded):
    """
    Performs feature engineering on the input DataFrame by creating a new categorical feature 'AGE_GROUP'
    based on the 'AGE_ABOVE65' column.

    Args:
        df_encoded (pd.DataFrame): The input DataFrame containing encoded features.

    Returns:
        pd.DataFrame: The DataFrame with the additional 'AGE_GROUP' feature.
    """
    logging.info("Performing feature engineering...")

    # Create a duplicate of the DataFrame to avoid modifying the original
    df_encoded_copy = df_encoded.copy()

    if df_encoded_copy.empty:
        logging.warning("DataFrame is empty before feature engineering.")
        return df_encoded_copy

    # Feature: Age Group (categorical)
    df_encoded_copy['AGE_GROUP'] = pd.cut(df_encoded_copy['AGE_ABOVE65'],
                                          bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                          labels=False,
                                          right=False)
    logging.info("Feature engineering completed.")
    return df_encoded_copy

    logging.info("Performing feature engineering...")
    # Create a duplicate of the DataFrame to avoid modifying the original
    df_encoded_copy = df_encoded.copy()
    if df_encoded_copy.empty:
        logging.warning("DataFrame is empty before feature engineering.")
        return df_encoded_copy
    # Feature: Age Group (categorical)
    df_encoded_copy['AGE_GROUP'] = pd.cut(df_encoded_copy['AGE_ABOVE65'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=False, right=False)
    logging.info("Feature engineering completed.")
    return df_encoded_copy

def encode_categorical_variables(df_encoded, categorical_cols):
    """
    Encodes categorical variables in the DataFrame using one-hot encoding.

    Args:
        df_encoded (pd.DataFrame): The input DataFrame containing encoded features.
        categorical_cols (list): List of categorical columns to be one-hot encoded.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded categorical variables.
    """
    logging.info("Encoding categorical variables...")

    if df_encoded.empty:
        logging.warning("DataFrame is empty before encoding.")

    # Initialize the OneHotEncoder with drop='first' to avoid multicollinearity
    encoder = OneHotEncoder(drop='first', sparse=False)
    
    # Perform one-hot encoding on the specified categorical columns
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols)

    print("After encoding:")
    print(df_encoded.head())
    return df_encoded

def normalize_numerical_features(df_encoded, numerical_cols):
    """
    Normalizes numerical features in the DataFrame using StandardScaler.

    Args:
        df_encoded (pd.DataFrame): The input DataFrame containing encoded features.
        numerical_cols (list): List of numerical columns to be normalized.

    Returns:
        pd.DataFrame: The DataFrame with normalized numerical features.
    """
    logging.info("Normalizing numerical features...")

    if df_encoded.empty:
        logging.warning("DataFrame is empty before normalization.")

    # Initialize the StandardScaler for normalization
    scaler = StandardScaler()

    # Create a copy to avoid modifying the original DataFrame
    df_encoded_copy = df_encoded.copy()

    # Normalize numerical columns using StandardScaler
    df_encoded_copy[numerical_cols] = scaler.fit_transform(df_encoded_copy[numerical_cols])

    print("After normalization:")
    print(df_encoded_copy.head())
    return df_encoded_copy

def data_exploration(df_encoded):
    """
    Performs exploratory data analysis (EDA) on the input DataFrame.

    Args:
        df_encoded (pd.DataFrame): The input DataFrame for exploratory data analysis.

    Returns:
        None
    """
    logging.info("Exploratory Data Analysis:")

    if df_encoded.empty:
        logging.warning("DataFrame is empty.")

    # Display dataset information
    print("Dataset Information:")
    print(df_encoded.info())

    # Display summary statistics
    print("\nSummary Statistics:")
    print(df_encoded.describe())

    # Plot the distribution of ICU Admission
    plt.figure(figsize=(6, 4))
    sns.countplot(x='ICU', data=df_encoded)
    plt.title('Distribution of ICU Admission')
    plt.show()

def perform_grid_search(X_train, y_train, clf, param_grid, model_type):
    """
    Performs grid search for hyperparameter tuning using cross-validation.

    Args:
        X_train (pd.DataFrame): The input features of the training set.
        y_train (pd.Series): The target variable of the training set.
        clf (object): The classifier or model for which hyperparameter tuning is performed.
        param_grid (dict): The grid of hyperparameters to search over.
        model_type (str): The type of model being tuned (e.g., 'xgboost').

    Returns:
        object: The best estimator/model after grid search.
    """
    logging.info(f"Performing grid search for {model_type} model...")

    # Initialize GridSearchCV with 5-fold cross-validation and accuracy as the scoring metric
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=6)
    
    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best hyperparameters
    best_params = grid_search.best_params_
    logging.info(f"Best Parameters ({model_type}): {best_params}")
    
    # Return the best estimator/model
    return grid_search.best_estimator_

def train_model(X_train, y_train, model_type='xgboost', param_grid=None, selected_features=None):
    """
    Trains a machine learning model using the specified model type, optionally with hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): The input features of the training set.
        y_train (pd.Series): The target variable of the training set.
        model_type (str): The type of model to train (e.g., 'xgboost').
        param_grid (dict, optional): The grid of hyperparameters for grid search (default: None).
        selected_features (list, optional): The list of selected features to use in training (default: None).

    Returns:
        object: The trained machine learning model.
    """
    logging.info(f"Training {model_type.capitalize()} model (this may take a while)...")

    # If selected features are specified, filter the training features
    if selected_features is not None:
        X_train = X_train[selected_features]

    # Initialize the XGBoost classifier with default parameters
    if model_type == 'xgboost':
        clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        default_param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
    else:
        raise ValueError("Invalid model type. Supported types: 'xgboost'")

    # Use the provided hyperparameter grid or default grid
    if param_grid is None:
        param_grid = default_param_grid

    # Remove 'min_samples_leaf' and 'min_samples_split' from param_grid
    param_grid = {key: value for key, value in param_grid.items() if key not in ['min_samples_leaf', 'min_samples_split']}

    # Perform grid search for hyperparameter tuning
    best_estimator = perform_grid_search(X_train, y_train, clf, param_grid, model_type)
    
    return best_estimator

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates the performance of a machine learning model using various metrics and visualizations.

    Args:
        model (object): The trained machine learning model.
        X_test (pd.DataFrame): The input features of the test set.
        y_test (pd.Series): The true target variable of the test set.
        threshold (float, optional): The threshold for converting probabilities to binary predictions (default: 0.5).

    Returns:
        None
    """
    logging.info("Evaluating model with threshold adjustment...")

    # Predict probabilities on the test set
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to binary predictions based on the threshold
    y_pred_adjusted = (y_pred_prob > threshold).astype(int)

    # Convert y_test to binary values (assuming it's continuous)
    y_test_binary = (y_test > 0.5).astype(int)

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test_binary, y_pred_adjusted))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_binary, y_pred_adjusted))

    # Print accuracy using the threshold-adjusted predictions
    accuracy = accuracy_score(y_test_binary, y_pred_adjusted)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(y_test_binary, y_pred_adjusted)
    print(f"Precision: {precision:.4f}")

    recall = recall_score(y_test_binary, y_pred_adjusted)
    print(f"Recall: {recall:.4f}")

    f1 = f1_score(y_test_binary, y_pred_adjusted)
    print(f"F1-score: {f1:.4f}")

    auc_roc = roc_auc_score(y_test_binary, y_pred_prob)
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test_binary, y_pred_adjusted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

def identify_column_type(df_current):
    """
    Identifies the data type of each column in a DataFrame and categorizes them as either numeric or categorical.

    Args:
        df_current (pd.DataFrame): The DataFrame for which column types need to be identified.

    Returns:
        tuple: A tuple containing two lists - categorical_cols and numerical_cols.
               - categorical_cols (list): List of column names with categorical data.
               - numerical_cols (list): List of column names with numerical data.
    """
    categorical_cols = []
    numerical_cols = []

    # Loop through each column in the DataFrame
    for column in df_current.columns:
        # Check if the data type of the column is numeric
        if pd.api.types.is_numeric_dtype(df_current[column]):
            numerical_cols.append(column)
        else:
            categorical_cols.append(column)

    # Return the identified categorical and numerical columns
    return categorical_cols, numerical_cols

def data_preprocessing(df):
    """
    Perform preprocessing on the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw data.

    Returns:
        pd.DataFrame: The processed DataFrame after applying various preprocessing steps.
    """
    logging.info("Preprocessing data...")

    # Apply the function to 'AGE_PERCENTIL' column
    df['AGE_PERCENTIL'] = df['AGE_PERCENTIL'].apply(extract_percentile)
    
    # Create the 'AGE_GROUP' feature
    df['AGE_GROUP'] = df['AGE_PERCENTIL'].apply(define_age_group)
    
    # Encode the 'WINDOW' column using ordinal encoding
    ordinal_encoder = OrdinalEncoder(categories=[['0-2', '2-4', '4-6', '6-12', 'ABOVE_12']])
    df[['WINDOW']] = ordinal_encoder.fit_transform(df[['WINDOW']])
    
    # Drop the 'AGE_PERCENTIL' column if no longer needed
    df.drop('AGE_PERCENTIL', axis=1, inplace=True)
    
    # Define categorical and numerical columns
    categorical_cols, numerical_cols = identify_column_type(df)
    
    # Handling Missing Values
    df = handle_missing_values(df, numerical_cols)
    
    # Convert the 'ICU' column to categorical
    df['ICU'] = df['ICU'].astype(int)
    
    # Feature Engineering
    df_encoded = feature_engineering(df)
    
    # Identify and remove constant features
    constant_features = df_encoded.columns[df_encoded.nunique() == 1]
    df_encoded = df_encoded.drop(columns=constant_features)
    
    # Redefine categorical and numerical columns for the updated dataframe
    categorical_cols_updated, numerical_cols_updated = identify_column_type(df_encoded)
    
    # Encoding Categorical Variables
    df_encoded = encode_categorical_variables(df_encoded, categorical_cols_updated)
    
    # Normalizing Numerical Features
    df_encoded = normalize_numerical_features(df_encoded, numerical_cols_updated)
    
    return df_encoded
    logging.info("Preprocessing data...")
    # Apply the function to 'AGE_PERCENTIL' column
    df['AGE_PERCENTIL'] = df['AGE_PERCENTIL'].apply(extract_percentile)
    # Create the 'AGE_GROUP' feature
    df['AGE_GROUP'] = df['AGE_PERCENTIL'].apply(define_age_group)
    # Encode the 'WINDOW' column using ordinal encoding
    ordinal_encoder = OrdinalEncoder(categories=[['0-2', '2-4', '4-6', '6-12', 'ABOVE_12']])
    df[['WINDOW']] = ordinal_encoder.fit_transform(df[['WINDOW']])
    # Drop the 'AGE_PERCENTIL' column if no longer needed
    df.drop('AGE_PERCENTIL', axis=1, inplace=True)
    # Define categorical and numerical columns
    categorical_cols, numerical_cols = identify_column_type(df)
    # Handling Missing Values
    df = handle_missing_values(df, numerical_cols)
    # Convert the 'ICU' column to categorical
    df['ICU'] = df['ICU'].astype(int)
    # Feature Engineering
    df_encoded = feature_engineering(df)
    # Identify and remove constant features
    constant_features = df_encoded.columns[df_encoded.nunique() == 1]
    df_encoded = df_encoded.drop(columns=constant_features)
    # Redefine categorical and numerical columns for updated dataframe
    categorical_cols_updated, numerical_cols_updated = identify_column_type(df_encoded)
    # Encoding Categorical Variables
    df_encoded = encode_categorical_variables(df_encoded, categorical_cols_updated)
    # Normalizing Numerical Features
    df_encoded = normalize_numerical_features(df_encoded, numerical_cols_updated)
    return df_encoded

def identify_important_features(df_encoded, n_features_to_select_range=range(1, 21), threshold=0.5):
    """
    Identify important features using Recursive Feature Elimination (RFE).

    Args:
        df_encoded (pd.DataFrame): The DataFrame containing encoded and preprocessed data.
        n_features_to_select_range (range, optional): Range of the number of features to select in RFE. Defaults to range(1, 21).
        threshold (float, optional): Threshold for converting labels to binary. Defaults to 0.5.

    Returns:
        pd.Index: The selected important features based on RFE.
    """
    logging.info("Identifying important features using Recursive Feature Elimination (RFE)...")

    # After handling class imbalance and before feature engineering, remove rows with missing values
    categorical_cols, numerical_cols = identify_column_type(df_encoded)
    df_no_missing = handle_missing_values(df_encoded, numerical_cols)

    # Splitting the Data for initial training
    X_task2_no_missing = df_no_missing.drop(['ICU'], axis=1)
    y_task2_no_missing = df_no_missing['ICU']

    # Convert to binary labels based on a threshold
    y_task2_no_missing_binary = (y_task2_no_missing > threshold).astype(int)

    X_train_no_missing_task2, X_test_no_missing_task2, y_train_no_missing_task2, y_test_no_missing_task2 = train_test_split(
        X_task2_no_missing, y_task2_no_missing_binary, test_size=0.2, random_state=42)

    # Handling Class Imbalance for the cleaned dataset
    X_train_resampled_task2, y_train_resampled_task2 = handle_class_imbalance(
        X_train_no_missing_task2, y_train_no_missing_task2)

    # Initialize the XGBClassifier for RFE
    clf_xgb_rfe = XGBClassifier(random_state=42)

    # Initialize RFE with the XGBClassifier
    rfe = RFE(estimator=clf_xgb_rfe, n_features_to_select=1)  # Start with 1 feature

    # Evaluate performance for different numbers of features using parallelized cross-validation
    cv_scores = []
    with parallel_backend('loky', n_jobs=-1):  # Use all available CPUs
        for n_features_to_select in n_features_to_select_range:
            rfe.n_features_to_select = n_features_to_select
            scores = cross_val_score(rfe, X_train_resampled_task2, y_train_resampled_task2, cv=5, scoring='accuracy')
            cv_scores.append(scores.mean())

    # Choose the number of features that maximizes the cross-validation score
    optimal_n_features = n_features_to_select_range[cv_scores.index(max(cv_scores))]

    # Fit RFE with the optimal number of features
    rfe.n_features_to_select = optimal_n_features
    rfe.fit(X_train_resampled_task2, y_train_resampled_task2)

    # Get the selected features
    selected_features_rfe = X_train_resampled_task2.columns[rfe.support_]

    return selected_features_rfe

    logging.info("Identifying important features using Recursive Feature Elimination (RFE)...")
    # After handling class imbalance and before feature engineering, remove rows with missing values
    categorical_cols, numerical_cols = identify_column_type(df_encoded)
    df_no_missing = handle_missing_values(df_encoded, numerical_cols)
    # Splitting the Data for initial training
    X_task2_no_missing = df_no_missing.drop(['ICU'], axis=1)
    y_task2_no_missing = df_no_missing['ICU']
    # Convert to binary labels based on a threshold
    y_task2_no_missing_binary = (y_task2_no_missing > threshold).astype(int)
    X_train_no_missing_task2, X_test_no_missing_task2, y_train_no_missing_task2, y_test_no_missing_task2 = train_test_split(
        X_task2_no_missing, y_task2_no_missing_binary, test_size=0.2, random_state=42)
    # Handling Class Imbalance for the cleaned dataset
    X_train_resampled_task2, y_train_resampled_task2 = handle_class_imbalance(
        X_train_no_missing_task2, y_train_no_missing_task2)
    # Initialize the XGBClassifier for RFE
    clf_xgb_rfe = XGBClassifier(random_state=42)
    # Initialize RFE with the XGBClassifier
    rfe = RFE(estimator=clf_xgb_rfe, n_features_to_select=1)  # Start with 1 feature
    # Evaluate performance for different numbers of features using parallelized cross-validation
    cv_scores = []
    with parallel_backend('loky', n_jobs=-1):  # Use all available CPUs
        for n_features_to_select in n_features_to_select_range:
            rfe.n_features_to_select = n_features_to_select
            scores = cross_val_score(rfe, X_train_resampled_task2, y_train_resampled_task2, cv=5, scoring='accuracy')
            cv_scores.append(scores.mean())
    # Choose the number of features that maximizes the cross-validation score
    optimal_n_features = n_features_to_select_range[cv_scores.index(max(cv_scores))]
    # Fit RFE with the optimal number of features
    rfe.n_features_to_select = optimal_n_features
    rfe.fit(X_train_resampled_task2, y_train_resampled_task2)
    # Get the selected features
    selected_features_rfe = X_train_resampled_task2.columns[rfe.support_]
    return selected_features_rfe

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Load the dataset
data = "COVID-Full.csv"
try:
    df = pd.read_csv(data)
except FileNotFoundError:
    logging.error(f"Error: Data file '{data}' not found.")
    exit(1)
except pd.errors.EmptyDataError:
    logging.error(f"Error: Data file '{data}' is empty.")
    exit(1)
except pd.errors.ParserError:
    logging.error(f"Error: Unable to parse data from '{data}'. Check the file format.")
    exit(1)

# Data Preprocessing
df_encoded = data_preprocessing(df)

# Identify important features
important_features = identify_important_features(df_encoded)

# Impute rows with missing values for important features from the original dataset
df_cleaned = handle_missing_values(df_encoded, important_features)

# Splitting the Data for updated training
X_task2 = df_cleaned.drop(['ICU'], axis=1)
y_task2 = df_cleaned['ICU']
X_train_task2, X_test_task2, y_train_task2, y_test_task2 = train_test_split(
    X_task2, y_task2, test_size=0.2, random_state=42)

# Handling Class Imbalance for the updated dataset
X_train_resampled_task2, y_train_resampled_task2 = handle_class_imbalance(X_train_task2, y_train_task2)

# Train the XGBoost model on the updated data with selected features
clf_xgb_tuned_task2 = train_model(
    X_train_resampled_task2, y_train_resampled_task2, model_type='xgboost', selected_features=important_features)

# Filter X_test_task2
X_test_task2_filtered = X_test_task2[important_features]

# Ensure y_test_task2 aligns with the filtered X_test_task2
y_test_task2_filtered = y_test_task2

# Evaluate the model
evaluate_model(clf_xgb_tuned_task2, X_test_task2_filtered, y_test_task2_filtered)