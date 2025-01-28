from sklearn.inspection import permutation_importance
import pandas as pd

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


def calculate_permutation_importance(model_name, feature_sets, X_test, y_test):
    """
    Calculates permutation importance for selected models and features.

    Parameters:
    - models: Dictionary of selected models (e.g., {"ModelName": model_object}).
    - feature_sets: Dictionary of feature sets corresponding to each model.
    - X_train: Training features (DataFrame).
    - y_train: Training target (Series or array).
    - X_test: Test features (DataFrame).
    - y_test: Test target (Series or array).

    Returns:
    - DataFrame: Aggregated permutation importance for each model.
    """
    # Initialize an empty DataFrame to store permutation importances
    perm_df = pd.DataFrame()



    # Get the features specific to the model
    selected_features = feature_sets[model_name]
    X_test_selected = X_test[selected_features]

    # Calculate permutation importance
    perm = permutation_importance(
        model,
        X_test_selected,
        y_test,
        n_repeats=100,  # Number of shuffles
        random_state=42,
        scoring="f1",  # Adjust scoring metric as needed
    )

    # Create a DataFrame for the model's permutation importance
    perm_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance Mean": perm.importances_mean
    })




    perm_df.sort_values(by="Average_Importance", ascending=False, inplace=True)

    print(f"Permutation importance calculation completed. Total features: {perm_df.shape[0]}.")

    return perm_df