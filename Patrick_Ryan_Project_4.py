import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import logging
from joblib import parallel_backend

# Set up logging
logging.basicConfig(level=logging.INFO)

def feature_engineering(df):
    logging.info("Performing feature engineering...")
    # Extract percentile from 'AGE_PERCENTIL'
    df['AGE_PERCENTIL'] = df['AGE_PERCENTIL'].apply(lambda x: 90 if 'Above' in x else int(x.replace('th', '')))
    # Define age group based on the extracted percentile
    df['AGE_GROUP'] = pd.cut(df['AGE_PERCENTIL'], bins=[0, 18, 35, 50, 65, np.inf], labels=['child', 'young_adult', 'middle_adult', 'older_adult', 'elderly'])
    # Encode the 'WINDOW' column using ordinal encoding
    ordinal_encoder = OrdinalEncoder(categories=[['0-2', '2-4', '4-6', '6-12', 'ABOVE_12']])
    df[['WINDOW']] = ordinal_encoder.fit_transform(df[['WINDOW']])
    # Drop the 'AGE_PERCENTIL' column if no longer needed
    df.drop('AGE_PERCENTIL', axis=1, inplace=True)
    return df

def handle_missing_values(df, numerical_cols, important_features=None):
    logging.info("Handling missing values...") 
    # Check if any important features have missing values
    if important_features:
        logging.info("Removing rows with missing values for important features...")
        missing_cols = [col for col in important_features if col in df.columns and df[col].isnull().any()]
        if missing_cols:
            # Remove rows with missing values for important features
            df_cleaned = df.dropna(subset=missing_cols)
            logging.info(f"Removed rows with missing values for important features.")
        else:
            logging.info("No rows with missing values for important features. DataFrame remains unchanged.")
            df_cleaned = df.copy()
    else:
        logging.info("Removing all rows with missing values...")
        df_cleaned = df.dropna()
        logging.info(f"Removed rows with missing values.")
        
    # Convert numerical_cols to pandas Index
    numerical_cols = pd.Index(numerical_cols)
    
    # Impute missing values for numerical columns
    if not numerical_cols.empty:
        logging.info("Imputing missing values for numerical columns...")
        imputer_numeric = SimpleImputer(strategy='median')
        df_cleaned.loc[:, numerical_cols] = imputer_numeric.fit_transform(df_cleaned[numerical_cols])
    
    # Impute missing values for categorical columns
    for col in df_cleaned.columns:
        if col not in numerical_cols:
            df_cleaned.loc[:, col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    return df_cleaned


def handle_class_imbalance(X_train, y_train):
    logging.info("Handling class imbalance...")
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

def encode_categorical_variables(df_encoded, categorical_cols):
    logging.info("Encoding categorical variables...")
    if df_encoded.empty: logging.warning("DataFrame is empty before encoding.")
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols)
    logging.info("After encoding:")
    logging.info(df_encoded.head())
    return df_encoded

def normalize_numerical_features(df_encoded, numerical_cols):
    logging.info("Normalizing numerical features...")
    if df_encoded.empty: logging.warning("DataFrame is empty before normalization.")
    scaler = StandardScaler()
    df_encoded_copy = df_encoded.copy()  # Create a copy to avoid modifying the original DataFrame
    df_encoded_copy[numerical_cols] = scaler.fit_transform(df_encoded_copy[numerical_cols])
    print("After normalization:")
    print(df_encoded_copy.head())
    return df_encoded_copy

def data_exploration(df_encoded):
    logging.info("Exploratory Data Analysis:")
    if df_encoded.empty: logging.warning("DataFrame is empty.")
    print("Dataset Information:")
    print(df_encoded.info())
    print("\nSummary Statistics:")
    print(df_encoded.describe())
    plt.figure(figsize=(6, 4))
    sns.countplot(x='ICU', data=df_encoded)
    plt.title('Distribution of ICU Admission')
    plt.show()

def perform_grid_search(X_train, y_train, clf, param_grid, model_type):
    logging.info(f"Performing grid search for {model_type} model...")
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=6)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    logging.info(f"Best Parameters ({model_type}): {best_params}")
    return grid_search.best_estimator_

def train_model(X_train, y_train, model_type='xgboost', param_grid=None, selected_features=None):
    logging.info(f"Training {model_type.capitalize()} model (this may take a while)...")
    if selected_features is not None:
        X_train = X_train[selected_features]
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
    if param_grid is None:
        param_grid = default_param_grid
    # Remove 'min_samples_leaf' and 'min_samples_split' from param_grid
    param_grid = {key: value for key, value in param_grid.items() if key not in ['min_samples_leaf', 'min_samples_split']}
    best_estimator = perform_grid_search(X_train, y_train, clf, param_grid, model_type)
    return best_estimator, selected_features

def plot_feature_importance(model, feature_names):
    logging.info("Visualizing feature importance...")
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_feature_names = np.array(feature_names)[sorted_idx]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], align="center")
    plt.xticks(range(len(feature_importance)), sorted_feature_names, rotation=90)
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()

def evaluate_model(model, X_test, y_test, thresholds=[0.4, 0.5, 0.6], feature_names=None):
    logging.info("Evaluating model with threshold adjustment...")
    best_f1_score = 0
    best_threshold = 0.5  # Default threshold
    for threshold in thresholds:
        # Get predicted probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        # Convert probabilities to binary predictions based on the threshold
        y_pred_adjusted = (y_pred_prob > threshold).astype(int)
        # Convert y_test to binary values (assuming it's continuous)
        y_test_binary = (y_test > 0.5).astype(int)
        # Calculate F1-score
        f1 = f1_score(y_test_binary, y_pred_adjusted)
        # Update the best threshold if current F1-score is better
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold
    # Log the best threshold
    logging.info(f"Best threshold for maximum F1-score: {best_threshold}")
    # Use the best threshold for final predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_adjusted = (y_pred_prob > best_threshold).astype(int)
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
    # Plot Feature Importance
    if feature_names is not None:
        plot_feature_importance(model, feature_names)

def identify_column_type(df_current):
    categorical_cols = []
    numerical_cols = []
    for column in df_current.columns:
        if pd.api.types.is_numeric_dtype(df_current[column]):
            numerical_cols.append(column)
        else:
            categorical_cols.append(column)
    return categorical_cols, numerical_cols

def data_preprocessing(df):
    logging.info("Preprocessing data...")
    # Feature Engineering
    df = feature_engineering(df)
    # Identify categorical and numerical columns
    categorical_cols, numerical_cols = identify_column_type(df)

    # Convert the 'ICU' column to categorical
    df.loc[:, 'ICU'] = df['ICU'].astype(int)
    # Redefine categorical and numerical columns for the updated dataframe
    categorical_cols_updated, numerical_cols_updated = identify_column_type(df)
    # Encoding Categorical Variables
    df_encoded = encode_categorical_variables(df, categorical_cols_updated)
    # Normalizing Numerical Features
    df_encoded = normalize_numerical_features(df_encoded, numerical_cols_updated)
    return df_encoded

def identify_important_features(df_encoded, n_features_to_select_range=range(1, 21), threshold=0.5):
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
clf_xgb_tuned_task2, selected_features = train_model(
    X_train_resampled_task2, y_train_resampled_task2, model_type='xgboost', selected_features=important_features)
# Filter X_test_task2
X_test_task2_filtered = X_test_task2[important_features]
# Ensure y_test_task2 aligns with the filtered X_test_task2
# (assuming the order of samples is the same)
y_test_task2_filtered = y_test_task2
# Evaluate the model
evaluate_model(clf_xgb_tuned_task2, X_test_task2_filtered, y_test_task2_filtered, thresholds=[0.4, 0.5, 0.6], feature_names=selected_features)