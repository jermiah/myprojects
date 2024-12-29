# Data manipulation and analysis
import pandas as pd
import numpy as np

# For cross-validation
from sklearn.model_selection import KFold

# Miscellaneous
import math


def target_encode(df, column, target_column=None, encoding_mapping=None):
    """
    Apply target encoding for a single column based on the target column.

    Parameters:
    - df: DataFrame with data.
    - column: Column to encode.
    - target_column: Target column used for encoding.
    - encoding_mapping: Existing encoding mapping (if applying to test data).

    Returns:
    - DataFrame with target encoded column.
    - Updated encoding mapping.
    """
    df[column] = df[column].astype(str).str.strip()  # Ensure column is in string format

    if target_column:
        # Training phase: calculate class probabilities for the column
        class_probs = df.groupby(column)[target_column].value_counts(normalize=True).unstack(fill_value=0)
        encoding_mapping = class_probs.to_dict(orient='index')

        # Apply encoding to the column
        for target_class in class_probs.columns:
            df[f'{column}_{target_class}_prob'] = df[column].map(
                lambda x: encoding_mapping.get(x, {}).get(target_class, 0)
            )
        print(f"Simple Target Encoding applied to: {column} based on target '{target_column}'")

        # Drop the original column after encoding
        df.drop(columns=[column], inplace=True)

    else:
        # Test phase: apply existing mapping
        if encoding_mapping is None:
            raise ValueError("Encoding mapping is required for test data.")

        for target_class in encoding_mapping.get(next(iter(encoding_mapping))):
            df[f'{column}_{target_class}_prob'] = df[column].map(
                lambda x: encoding_mapping.get(x, {}).get(target_class, 0)
            )
        print(f"Simple Target Encoding applied to: {column} using pre-learned mapping")

        # Drop the original column after encoding
        df.drop(columns=[column], inplace=True)

    return df, encoding_mapping



def target_encode_with_cv(df, column, target_column, encoding_mapping=None, smoothing=1.0, cv_folds=5):
    """
    Apply regularized target encoding using cross-validation.

    Parameters:
    - df: DataFrame with data.
    - column: Column to encode.
    - target_column: Target column used for encoding.
    - encoding_mapping: Existing encoding mapping (if applying to test data).
    - smoothing: Regularization parameter for smoothing.
    - cv_folds: Number of folds for cross-validation.

    Returns:
    - DataFrame with target encoded column.
    - Updated encoding mapping.
    """
    df[column] = df[column].astype(str).str.strip()  # Ensure column is in string format

    if encoding_mapping is None:
        encoding_mapping = {}

    # Cross-validation setup
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Calculate the global mean of the target column
    global_mean = df[target_column].mean()

    # Calculate class probabilities for the column with regularization
    group_mean = df.groupby(column)[target_column].mean()
    group_count = df.groupby(column).size()

    # Apply smoothing (regularization)
    smoothed_mean = (group_mean * group_count + global_mean * smoothing) / (group_count + smoothing)
    encoding_mapping[column] = smoothed_mean.to_dict()

    # Apply encoding to the column
    df[f'{column}_encoded'] = df[column].map(
        lambda x: encoding_mapping[column].get(x, global_mean)
    )
    print(f"Regularized Target Encoding applied to: {column} based on target '{target_column}'")

    # Drop the original column after encoding
    df.drop(columns=[column], inplace=True)

    return df, encoding_mapping


def encode_data(df):
    """
    Encodes the DataFrame:
    - One-Hot Encoding for nominal variables.
    - Ordinal Encoding for specific columns.
    
    Parameters:
    - df: The input DataFrame.

    Returns:
    - Processed DataFrame with encoded features.
    """
    print("Encoding data...")

    # Define nominal and ordinal columns
    nominal_columns = ['hospital_type_code', 'city_code_hospital', 
                       'hospital_region_code', 'department', 'ward_type', 'ward_facility_code', 'bed_grade']

    severity_mapping = {'Minor': 1, 'Moderate': 2, 'Extreme': 3}
    age_mapping = {'0-10': 1, '11-20': 2, '21-30': 3, '31-40': 4, '41-50': 5, '51-60': 6, '61-70': 7, '71-80': 8, '81-90': 9, '91-100': 10}
    admission_mapping = {'Emergency': 1, 'Trauma': 2, 'Urgent': 3}
    stay_mapping = {'0-10': 1, '11-20': 2, '21-30': 3, '31-40': 4, '41-50': 5, '51-60': 6, '61-70': 7, '71-80': 8, '81-90': 9, '91-100': 10, 'More than 100 Days': 11}

    ordinal_columns_mappings = {
        'severity_of_illness': severity_mapping,
        'age': age_mapping,
        'type_of_admission': admission_mapping,
        'stay': stay_mapping,
    }

    # Apply ordinal encoding
    for col, mapping in ordinal_columns_mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(mapping)
            print(f"Ordinal Encoding applied to '{col}'")

    # Apply one-hot encoding for nominal columns
    nominal_columns_present = [col for col in nominal_columns if col in df.columns]
    if nominal_columns_present:
        df = pd.get_dummies(df, columns=nominal_columns_present, drop_first=True)
        print(f"One-Hot Encoding applied to: {', '.join(nominal_columns_present)}")

    # Return the processed DataFrame
    return df
