import pandas as pd


from sklearn.inspection import permutation_importance


def aggregate_feature_importance(models):
    """
    Aggregates feature importance from multiple models and standardizes feature names.

    Parameters:
    - models: Dictionary of model names and fitted model objects.

    Returns:
    - DataFrame: A table where rows are features, columns are model importance, 
                 and the last column is the average importance.
    """
    combined_df = pd.DataFrame()

    for model_name, model in models.items():
        # Handle CatBoost feature importance
        if hasattr(model, 'get_feature_importance') and hasattr(model, 'feature_names_'):
            importances = model.get_feature_importance()
            features = model.feature_names_
        elif hasattr(model, 'feature_importances_'):  # Tree-based models
            importances = model.feature_importances_
            features = getattr(model, 'feature_names_in_', range(len(importances)))  # Use model's feature names if available
        elif hasattr(model, 'coef_'):  # Linear models
            importances = abs(model.coef_[0])  # Use absolute values for coefficients
            features = range(len(importances))
        else:
            continue

        # Prepare the importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': features, 
            'Importance': importances
        })

        # Normalize importance values for this model
        importance_df['Normalized_Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
        importance_df = importance_df.rename(columns={'Normalized_Importance': model_name})

        # Merge with the combined DataFrame
        combined_df = pd.merge(
            combined_df, 
            importance_df[['Feature', model_name]], 
            on='Feature', 
            how='outer'
        ) if not combined_df.empty else importance_df[['Feature', model_name]]

    # Calculate the average importance across all models
    combined_df['Average_Importance'] = combined_df.drop(columns=['Feature']).mean(axis=1, skipna=True)
    combined_df = combined_df.sort_values(by='Average_Importance', ascending=False)

    return combined_df



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
