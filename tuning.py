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


# Detect device (CPU/GPU) for models
def detect_device(model_name):
    if model_name == "CatBoost":
        return "GPU" if CatBoostClassifier().get_device_type() == "GPU" else "CPU"
    elif model_name == "LightGBM":
        return "gpu" if LGBMClassifier(boosting_type="gbdt")._n_features_in_ else "cpu"
    elif model_name == "XGBoost":
        try:
            import xgboost
            if xgboost.rabit.get_world_size() > 1:
                return "GPU"
        except ImportError:
            pass
        return "CPU"
    return "CPU"


# Check system resources
def check_system_resources():
    cores = psutil.cpu_count(logical=True)  # Total logical cores
    memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
    print(f"System has {cores} CPU cores and {memory:.2f} GB of RAM.")
    return cores, memory


# Estimate training time
def estimate_training_time(model_name, n_samples, n_features, device):
    system_cores, memory = check_system_resources()

    # Rough scaling factors based on empirical observations
    scaling_factor = 1.0
    if device == "GPU":
        scaling_factor = 0.3  # GPUs are faster
    elif system_cores >= 8:  # Multi-core CPUs
        scaling_factor = 0.7
    else:
        scaling_factor = 1.2  # Single-core or resource-limited environments

    if model_name in ["CatBoost", "LightGBM", "XGBoost"]:
        return n_samples * n_features * (np.log(n_samples) if n_samples > 0 else 1) * scaling_factor * 1e-6
    else:
        return n_samples * n_features * scaling_factor * 1e-6


# Model evaluation function
def evaluate_models(X_train_encoded, X_test_encoded, X_train_non_encoded, X_test_non_encoded, y_train, y_test, models, numerical_columns, prev_results_df=None, categorical_columns=None):
    scaler_log_reg_svm = StandardScaler()  # Scaler for logistic regression and SVM
    results = []  # Initialize results list
    trained_models = {}  # Dictionary to store trained models

    n_samples, n_features = X_train_encoded.shape  # Assume encoded dataset size

    for model_name, model in models.items():
        print(f"Starting {model_name}...")

        try:
            # Determine whether to use encoded or non-encoded data
            if model_name in ['CatBoost', 'LightGBM']:
                # Use the non-encoded dataset
                X_train_scaled, X_test_scaled = X_train_non_encoded.copy(), X_test_non_encoded.copy()
                device = detect_device(model_name)
                print(f"Using non-encoded dataset for {model_name} on {device} because it natively supports categorical features.")
                if model_name == 'CatBoost' and categorical_columns:
                    model.set_params(cat_features=categorical_columns)
            elif model_name in ['Logistic Regression', 'SVM']:
                # Use the encoded dataset and scale numerical features for linear models
                X_train_scaled = X_train_encoded.copy()
                X_test_scaled = X_test_encoded.copy()
                X_train_scaled[numerical_columns] = scaler_log_reg_svm.fit_transform(X_train_encoded[numerical_columns])
                X_test_scaled[numerical_columns] = scaler_log_reg_svm.transform(X_test_encoded[numerical_columns])
                device = "CPU"  # Linear models typically run on CPU
            else:
                # Use the encoded dataset for other models
                X_train_scaled, X_test_scaled = X_train_encoded.copy(), X_test_encoded.copy()
                device = detect_device(model_name)

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
            print(f"Finished {model_name} in {elapsed_time:.2f} seconds.\n")

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
