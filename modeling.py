import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import pandas as pd
import os
import numpy as np

def estimate_training_time(model_name, n_samples, n_features, device):
    """
    Estimate training time based on model type, dataset size, and available hardware.
    Args:
        model_name (str): Name of the model (e.g., 'Random Forest', 'LightGBM', etc.).
        n_samples (int): Number of samples in the training dataset.
        n_features (int): Number of features in the training dataset.
        device (str): 'CPU' or 'GPU'.
    Returns:
        float: Estimated training time in seconds.
    """
    # Base time per sample-feature product (empirical values)
    base_time_per_sf = {
        "Random Forest": 1e-6,  # Empirical base time for Random Forest
        "Gradient Boosting": 2e-6,
        "LightGBM": 1e-7 if device == "GPU" else 5e-7,
        "XGBoost": 2e-7 if device == "GPU" else 1e-6,
        "CatBoost": 2e-7 if device == "GPU" else 1e-6,
        "Logistic Regression": 5e-8,
        "SVM": 1e-5,
    }

    # Get base time for the specified model
    base_time = base_time_per_sf.get(model_name, 1e-6)  # Default base time if model is not in the dictionary

    # Calculate the estimated time (linear approximation)
    estimated_time = base_time * n_samples * n_features

    # Add hardware-dependent scaling factors
    if device == "GPU":
        estimated_time *= 0.5  # Assume GPUs are twice as fast

    return estimated_time


# Check system resources and availability of GPU
def check_system_resources():
    cores = psutil.cpu_count(logical=True)  # Total logical cores
    memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
    
    # Check if GPU is available
    gpu_available = False
    try:
        import torch  # Using PyTorch for GPU detection
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    gpu_status = "GPU available" if gpu_available else "No GPU available"
    print(f"System has {cores} CPU cores, {memory:.2f} GB of RAM, and {gpu_status}.")
    return cores, memory, gpu_available

# Detect device (CPU/GPU) for models
def detect_device(model_name, gpu_available):
    if model_name in ["CatBoost", "LightGBM", "XGBoost"]:
        return "GPU" if gpu_available else "CPU"
    return "CPU"

# Model evaluation function
def evaluate_models(X_train_encoded, X_test_encoded, X_train_non_encoded, X_test_non_encoded, y_train, y_test, models, numerical_columns, prev_results_df=None, categorical_columns=None):
    # Check system resources (once per pipeline)
    cores, memory, gpu_available = check_system_resources()  # System check
    
    scaler_log_reg_svm = StandardScaler()  # Scaler for logistic regression and SVM
    results = []  # Initialize results list
    trained_models = {}  # Dictionary to store trained models

    n_samples, n_features = X_train_encoded.shape  # Assume encoded dataset size

    for model_name, model in models.items():
        # Detect device for the current model
        device = detect_device(model_name, gpu_available)
        print(f"\nStarting {model_name} on {device}...")

        try:
            # Determine whether to use encoded or non-encoded data
            if model_name in ['CatBoost', 'LightGBM']:
                # Use the non-encoded dataset
                X_train_scaled, X_test_scaled = X_train_non_encoded.copy(), X_test_non_encoded.copy()
                print(f"Using non-encoded dataset for {model_name}.")
                if model_name == 'CatBoost' and categorical_columns:
                    model.set_params(cat_features=categorical_columns)
            elif model_name in ['Logistic Regression', 'SVM']:
                # Use the encoded dataset and scale numerical features for linear models
                X_train_scaled = X_train_encoded.copy()
                X_test_scaled = X_test_encoded.copy()
                X_train_scaled[numerical_columns] = scaler_log_reg_svm.fit_transform(X_train_encoded[numerical_columns])
                X_test_scaled[numerical_columns] = scaler_log_reg_svm.transform(X_test_encoded[numerical_columns])
            else:
                # Use the encoded dataset for other models
                X_train_scaled, X_test_scaled = X_train_encoded.copy(), X_test_encoded.copy()

            # Estimate training time
            estimated_training_time = estimate_training_time(model_name, n_samples, n_features, device)
            print(f"Estimated training time for {model_name}: ~{estimated_training_time:.2f} seconds.")

            # Measure actual training time
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            elapsed_time = time.time() - start_time
            print(f"Actual training time for {model_name}: {elapsed_time:.2f} seconds.")

            # Save the trained model
            trained_models[model_name] = model  # Store the trained model in the dictionary

            # Make predictions on training and testing data
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Handle probabilities (for models supporting `predict_proba`)
            if hasattr(model, 'predict_proba'):
                y_test_pred_prob = model.predict_proba(X_test_scaled)
                roc_auc_test = roc_auc_score(y_test, y_test_pred_prob, multi_class="ovr")  # Multiclass ROC AUC for test
            else:
                roc_auc_test = None

            # Calculate metrics for test set
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average="weighted")
            test_recall = recall_score(y_test, y_test_pred, average="weighted")
            test_f1 = f1_score(y_test, y_test_pred, average="weighted")

            # Calculate metrics for training set
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average="weighted")
            train_recall = recall_score(y_train, y_train_pred, average="weighted")
            train_f1 = f1_score(y_train, y_train_pred, average="weighted")

            results.append({
                "Model": model_name,
                "Train Accuracy": train_accuracy,
                "Train Precision": train_precision,
                "Train Recall": train_recall,
                "Train F1 Score": train_f1,
                "Test Accuracy": test_accuracy,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test F1 Score": test_f1,
                "Test ROC AUC": roc_auc_test,
                "Estimated Time (s)": estimated_training_time,
                "Actual Time (s)": elapsed_time,
            })
            print(f"Finished {model_name}.\n")

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue  # Skip to the next model

    # Create a final DataFrame for the results
    results_df = pd.DataFrame(results)

    # Append new results to the previous results DataFrame, if provided
    if prev_results_df is not None:
        results_df_all = pd.concat([results_df, prev_results_df], axis=0, ignore_index=True)
    else:
        results_df_all = results_df

    return results_df_all, trained_models  # Return both results and trained models


def run_full_pipeline(
    X_train_encoded, X_test_encoded, X_train_non_encoded, X_test_non_encoded,
    y_train, y_test, numerical_columns, categorical_columns=None, prev_results_df=None
):
    print("Function run_full_pipeline reloaded successfully.")
    # Display system resources and determine GPU availability
    cores, memory, gpu_available = check_system_resources()

    # Adjust target labels to start from 0 if needed (for multiclass compatibility)
    y_train_shifted = y_train - 1
    y_test_shifted = y_test - 1

    # Debug categorical columns
    if not isinstance(categorical_columns, list):
        categorical_columns = list(categorical_columns)  # Ensure it is a list

    # Remove 'stay' if it exists in categorical_columns
    if 'stay' in categorical_columns:
        categorical_columns.remove('stay')
        print("Removed 'stay' from categorical_columns.")

    # Ensure categorical columns exist in the dataset
    valid_columns = [col for col in categorical_columns if col in X_train_non_encoded.columns.tolist()]

    # Convert valid categorical columns to `category` dtype
    for col in valid_columns:
        X_train_non_encoded[col] = X_train_non_encoded[col].astype("category")
        X_test_non_encoded[col] = X_test_non_encoded[col].astype("category")

    # Define the models to evaluate
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(
            n_jobs=-1, 
            random_state=42, 
            device='gpu' if detect_device("LightGBM", gpu_available) == "GPU" else 'cpu',
            boosting_type='gbdt',
            categorical_feature=categorical_columns  # Pass categorical columns
        ),
        'XGBoost': xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1,
            tree_method='gpu_hist' if detect_device("XGBoost", gpu_available) == "GPU" else 'auto'
        ),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(
            verbose=0, 
            random_state=42, 
            task_type='GPU' if detect_device("CatBoost", gpu_available) == "GPU" else 'CPU'
        ),
        'Logistic Regression': OneVsRestClassifier(
            LogisticRegression(
                solver='saga', max_iter=500, random_state=42, n_jobs=-1
            )
        ),
        'SVM': OneVsRestClassifier(
            LinearSVC(random_state=42, max_iter=5000)
        ),
    }

    # Debug the models dictionary
    print(f"Models defined: {list(models.keys())}")

    # Run the evaluation function
    results_df_all_models, trained_models = evaluate_models(
        X_train_encoded=X_train_encoded,
        X_test_encoded=X_test_encoded,
        X_train_non_encoded=X_train_non_encoded,
        X_test_non_encoded=X_test_non_encoded,
        y_train=y_train_shifted,
        y_test=y_test_shifted,
        models=models,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        prev_results_df=prev_results_df,
    )

    return results_df_all_models, trained_models
