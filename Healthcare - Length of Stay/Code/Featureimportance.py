import pandas as pd


from sklearn.inspection import permutation_importance

def aggregate_feature_importance(models, X_train_encoded, X_train_non_encoded, selected_models):
    """
    Aggregates feature importance from selected trained models.

    Parameters:
    - models: Dictionary of trained model names and objects.
    - X_train_encoded: Encoded training data (for models that require numeric input).
    - X_train_non_encoded: Non-encoded training data (for models that handle categorical features natively).
    - selected_models: List of model names for which feature importance should be computed.

    Returns:
    - Tuple[DataFrame, DataFrame]: Two DataFrames for aggregated feature importances:
        - One for models using encoded data.
        - One for models using non-encoded data.
    """
    import pandas as pd

    # Initialize DataFrames for encoded and non-encoded models
    encoded_df, non_encoded_df = pd.DataFrame(), pd.DataFrame()

    for model_name, model in models.items():
        if model_name not in selected_models:
            continue

        print(f"Processing feature importance for: {model_name}")

        # Select appropriate dataset
        X_train = X_train_non_encoded if model_name in ['CatBoost', 'LightGBM'] else X_train_encoded
        combined_df = non_encoded_df if model_name in ['CatBoost', 'LightGBM'] else encoded_df

        # Extract feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            importances = model.get_feature_importance()
        elif hasattr(model, "coef_"):
            importances = abs(model.coef_[0])
        else:
            print(f"Skipping {model_name}: No feature importance available.")
            continue

        # Validate feature importance length
        if len(importances) != X_train.shape[1]:
            print(f"Skipping {model_name}: Feature importance length ({len(importances)}) "
                  f"does not match number of features ({X_train.shape[1]}).")
            continue

        # Create a DataFrame for the model's feature importance
        importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            model_name: importances / importances.sum() if importances.sum() != 0 else 0
        })

        # Merge with the appropriate combined DataFrame
        combined_df = pd.merge(combined_df, importance_df, on="Feature", how="outer") if not combined_df.empty else importance_df

        # Update the respective DataFrame
        if model_name in ['CatBoost', 'LightGBM']:
            non_encoded_df = combined_df
        else:
            encoded_df = combined_df

    # Add average importance column to both DataFrames
    for df_name, df in [("Encoded DataFrame", encoded_df), ("Non-Encoded DataFrame", non_encoded_df)]:
        if not df.empty:
            df["Average_Importance"] = df.drop(columns=["Feature"]).mean(axis=1, skipna=True)
            df.sort_values(by="Average_Importance", ascending=False, inplace=True)
            print(f"{df_name} processed. Total features: {df.shape[0]}.")

    return encoded_df, non_encoded_df



def compute_permutation_importance(selected_models, model_specific_features, X_train, y_train):
    """
    Computes permutation importance for multiple models using their pre-selected features.

    Parameters:
    - selected_models: dict of trained models (e.g., {"Gradient Boosting": model1, ...}).
    - model_specific_features: dict of selected features for each model (e.g., {"Gradient Boosting": feature_list1, ...}).
    - X_train: DataFrame containing the training features.
    - y_train: Series or array containing the training target variable.

    Returns:
    - permutation_results: dict of DataFrames with permutation importance scores for each model.
    """
    # Initialize a dictionary to store results
    permutation_results = {}

    # Compute permutation importance for each model
    for model_name, model in selected_models.items():
        print(f"Computing permutation importance for {model_name}...")
        
        # Get the features specific to the current model
        specific_features = model_specific_features[model_name]
        X_train_filtered = X_train[specific_features]  # Filter training data to these features
        
        # Compute permutation importance
        perm_importance = permutation_importance(
            model, X_train_filtered, y_train, scoring="accuracy", random_state=42
        )
        
        # Create a DataFrame for the results
        perm_importance_df = pd.DataFrame({
            "Feature": X_train_filtered.columns,
            "Importance": perm_importance.importances_mean
        }).sort_values(by="Importance", ascending=False)
        
        # Store the results in the dictionary
        permutation_results[model_name] = perm_importance_df

        # Print the top features for this model
        print(f"Top features for {model_name} based on permutation importance:\n{perm_importance_df}")

    return permutation_results
