
# Suppress warnings
warnings.filterwarnings("ignore")
def evaluate_models(X_train, X_test, y_train, y_test, models, numerical_columns, prev_results_df=None):
    scaler_log_reg_svm = StandardScaler()  # Scaler for logistic regression and SVM
    results = []  # Initialize results list
    trained_models = {}  # Dictionary to store trained models

    for model_name, model in models.items():
        print(f"Starting {model_name}...")
        start_time = time.time()

        try:
            # Scale data for linear models (Logistic Regression and SVM)
            if model_name in ['Logistic Regression', 'SVM']:
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                X_train_scaled[numerical_columns] = scaler_log_reg_svm.fit_transform(X_train[numerical_columns])
                X_test_scaled[numerical_columns] = scaler_log_reg_svm.transform(X_test[numerical_columns])
            else:
                X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()

            # Train the model
            model.fit(X_train_scaled, y_train)

            # Save the trained model
            trained_models[model_name] = model  # Store the trained model in the dictionary

            # Make predictions on training and testing data
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Handle probabilities (for models supporting `predict_proba`)
            if hasattr(model, 'predict_proba'):
                y_test_pred_prob = model.predict_proba(X_test_scaled)
                roc_auc_test = roc_auc_score(y_test, y_test_pred_prob, multi_class='ovr')  # Multiclass ROC AUC for test
            else:
                roc_auc_test = None

            # Calculate metrics for test set
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average='weighted')
            test_recall = recall_score(y_test, y_test_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            # Calculate metrics for training set
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average='weighted')
            train_recall = recall_score(y_train, y_train_pred, average='weighted')
            train_f1 = f1_score(y_train, y_train_pred, average='weighted')

            elapsed_time = time.time() - start_time
            results.append({
                'Model': model_name,
                'Train Accuracy': train_accuracy,
                'Train Precision': train_precision,
                'Train Recall': train_recall,
                'Train F1 Score': train_f1,
                'Test Accuracy': test_accuracy,
                'Test Precision': test_precision,
                'Test Recall': test_recall,
                'Test F1 Score': test_f1,
                'Test ROC AUC': roc_auc_test,
                'Time Taken (s)': elapsed_time
            })
            print(f"Finished {model_name} in {elapsed_time:.2f} seconds.\n")

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue  # Skip to the next model

        # Show intermediate results
        results_df = pd.DataFrame(results).sort_values(by='Time Taken (s)', ascending=True)
        print("\nIntermediate Results After Completing", model_name, ":\n")
        display(results_df)

    # Create a final DataFrame for the results
    results_df = pd.DataFrame(results)

    # Append new results to the previous results DataFrame, if provided
    if prev_results_df is not None:
        results_df_all = pd.concat([results_df, prev_results_df], axis=0, ignore_index=True)
    else:
        results_df_all = results_df
    
    return results_df_all, trained_models  # Return both results and trained models
def run_full_pipeline(X_train, X_test, y_train, y_test, numerical_columns, prev_results_df=None):
    
    shutil.rmtree('catboost_info', ignore_errors=True)
    # Adjust target labels to start from 0 if needed
    y_train_shifted = y_train - 1
    y_test_shifted = y_test - 1

    # Models reordered for efficiency with your dataset
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_jobs=-1, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
        'Logistic Regression': OneVsRestClassifier(LogisticRegression(solver='saga', max_iter=500, random_state=42, n_jobs=-1)),
        'SVM': OneVsRestClassifier(LinearSVC(random_state=42, max_iter=5000))
    }

    # Run model evaluation
    results_df_all_models, trained_models = evaluate_models(
        X_train, X_test, y_train_shifted, y_test_shifted, models, numerical_columns, prev_results_df
    )

    return results_df_all_models, trained_models

