import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder

def summarize_nulls_and_modalities(df, columns):
    
    # Calculate the count of null values for each specified column
    null_counts = df[columns].isnull().sum()
    null_counts = null_counts[null_counts > 0]  # Filter columns with non-zero null counts

    # Calculate the percentage of null values for each specified column
    null_percentage = df[columns].isnull().mean() * 100
    null_percentage = null_percentage[null_percentage > 0]  # Filter columns with non-zero null percentages

    # Calculate the number of unique values (modalities) for each specified column
    modalities = df[columns].nunique()

    # Combine counts, percentages, and modalities into a single summary DataFrame
    null_summary = pd.DataFrame({
        'Column Name': null_counts.index,
        'Null Count': null_counts.values,
        'Null Percentage': null_percentage.values,
        'Modalities': modalities[null_counts.index].values,
        'Data Type': df[null_counts.index].dtypes.values  # Get data types directly for columns with nulls
    })

    # Sort by null count in descending order
    null_summary = null_summary.sort_values(by='Null Count', ascending=False).reset_index(drop=True)

    return null_summary

def get_descriptions_for_variables(df, variables):
    # Select the first two columns by their positional indices
    subset_df = df.iloc[:, :2]  # This selects all rows and the first two columns

    # Filter the subset DataFrame where the first column matches the 'variables' list
    result_df = subset_df[subset_df.iloc[:, 0].isin(variables)]

    # Return the resulting DataFrame with reset index
    return result_df.reset_index(drop=True)

import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical_variables(data, categorical_variables, orientation='vertical'):

    num_vars = len(categorical_variables)
    cols = math.ceil(math.sqrt(num_vars))  # Calculate number of columns
    rows = math.ceil(num_vars / cols)     # Calculate number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    # Ensure axes is iterable
    if num_vars == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()  # Flatten axes array

    for i, variable in enumerate(categorical_variables):
        ax = axes[i]
        value_counts = data[variable].value_counts()
        

        colors = sns.color_palette("pastel", len(value_counts))

        if orientation == 'vertical':
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette=colors)
            ax.set_xlabel(variable)
            ax.set_ylabel('Count')

            # Add data labels on vertical bars
            max_value = value_counts.max()
            for index, value in enumerate(value_counts.values):
                ax.text(index, value, str(value), ha='center', va='bottom', fontsize=9)
            ax.set_ylim(0, max_value * 1.15)  # Add 15% space above the tallest bar
        elif orientation == 'horizontal':
            sns.barplot(y=value_counts.index, x=value_counts.values, ax=ax, palette=colors)
            ax.set_xlabel('Count')
            ax.set_ylabel(variable)

            # Add data labels on horizontal bars
            max_value = value_counts.max()
            for index, value in enumerate(value_counts.values):
                ax.text(value, index, str(value), ha='left', va='center', fontsize=9)
            ax.set_xlim(0, max_value * 1.15)  # Add 15% space beyond the longest bar
        else:
            raise ValueError("Orientation must be either 'vertical' or 'horizontal'")

        ax.set_title(f'Frequency of {variable}')
        ax.tick_params(axis='x', rotation=45 if orientation == 'vertical' else 0)

    # Remove unused axes
    if num_vars < len(axes):
        for j in range(num_vars, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()  # Automatically adjust subplot spacing
    plt.show()
    
def plot_numerical_variables(data, numerical_variables):
    """
    Plots histograms for numerical variables in a layout similar to the categorical variables plot.
    Includes a dynamic layout with data labels for each variable.
    
    Parameters:
        data (DataFrame): The dataset containing the variables.
        numerical_variables (list): List of numerical variable names to plot.
    """
    num_vars = len(numerical_variables)
    cols = math.ceil(math.sqrt(num_vars))  # Calculate number of columns
    rows = math.ceil(num_vars / cols)     # Calculate number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    # Ensure axes is iterable
    if num_vars == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()  # Flatten axes array for consistent iteration

    for i, variable in enumerate(numerical_variables):
        ax = axes[i]
        # Plot the histogram with KDE (kernel density estimate)
        sns.histplot(data[variable], kde=True, ax=ax, color='skyblue', alpha=0.7)

        ax.set_title(f'Distribution of {variable}')
        ax.set_xlabel(variable)
        ax.set_ylabel('Density')

        # Dynamically adjust y-axis to accommodate large values
        max_value = data[variable].max()
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # Add 15% padding to y-axis

    # Remove unused axes
    if num_vars < len(axes):
        for j in range(num_vars, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()  # Automatically adjust subplot spacing
    plt.show()
     
def plot_correlation_heatmap(data):
    """
    Plots the correlation heatmap for numerical variables in the dataset.
    """
    plt.figure(figsize=(10, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()



def create_pairplot(df, corner=True ):
    """
    Creates a Seaborn pairplot for feature relationships grouped by target classes.

    Parameters:
    - df: DataFrame, the dataset containing features and target column
    - target_col: str, the name of the target column for grouping
    - palette: str, color palette for the plot (default is 'Set2')
    - corner: bool, whether to show only the lower triangle of the plot (default is True)
    - title: str, title of the pairplot (default is "Pairplot of Features Grouped by Target Class")

    Returns:
    - None
    """
    sns.pairplot(df, corner=corner)

    plt.show()
    

def groupby_combined_plots(data, groupby_combinations, agg_columns=None, agg_func="count", plot_kind="bar", title_prefix=None, stacked=False):
    """
    Creates combined grouped plots for numerical and categorical columns.

    Parameters:
        data (DataFrame): The input DataFrame.
        groupby_combinations (list): List of lists specifying groupings (e.g., [["Stay", "Age"], ["Stay", "Severity of Illness"]]).
        agg_columns (list, optional): List of numerical columns to aggregate. If None, defaults to counting rows for categorical data.
        agg_func (str or function, optional): Aggregation function (e.g., "sum", "mean", "count"). Defaults to "count".
        plot_kind (str, optional): Type of plot (e.g., "bar", "line"). Defaults to "bar".
        title_prefix (str, optional): Prefix for the plot titles.
        stacked (bool, optional): Whether to stack the bars in the bar chart. Defaults to False.
    """
    num_combinations = len(groupby_combinations)
    num_agg_columns = len(agg_columns or ["Count"])

    cols = num_agg_columns
    rows = num_combinations

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)

    for row_idx, groupby_columns in enumerate(groupby_combinations):
        # Handle numerical or categorical cases
        if agg_columns is None:
            # Default to counting rows
            grouped_data = data.groupby(groupby_columns).size().reset_index(name="Count")
            agg_columns_to_use = ["Count"]
        else:
            grouped_data = data.groupby(groupby_columns).agg({col: agg_func for col in agg_columns}).reset_index()
            agg_columns_to_use = agg_columns

        for col_idx, agg_column in enumerate(agg_columns_to_use):
            ax = axes[row_idx, col_idx]

            # Pivot for multi-index groupings
            pivot_data = grouped_data.pivot(index=groupby_columns[0], columns=groupby_columns[1], values=agg_column) \
                if len(groupby_columns) > 1 else grouped_data.set_index(groupby_columns[0])[agg_column]

            # Plot the data
            if plot_kind == "bar" and isinstance(pivot_data, pd.DataFrame):
                pivot_data.plot(kind="bar", ax=ax, stacked=stacked, legend=True)
            elif isinstance(pivot_data, pd.Series):
                pivot_data.plot(kind=plot_kind, ax=ax, legend=False)

            # Set plot title and labels
            if agg_columns is None:
                title = f"Count of {groupby_columns[1]} by {groupby_columns[0]}"
            else:
                title = f"{title_prefix if title_prefix else ''}{agg_func.capitalize()} of {agg_column} by {' and '.join(groupby_columns)}"
            ax.set_title(title)
            ax.set_xlabel(groupby_columns[0])
            ax.set_ylabel(f"{agg_func.capitalize()} of {agg_column}" if agg_columns else "Count")

            if len(groupby_columns) > 1:
                ax.legend(title=groupby_columns[1], bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

def process_data(df):
    """
    Processes the DataFrame by:
    - Converting specified columns to categorical data type.
    - Dropping unnecessary columns ('case_id' and 'patientid').

    Parameters:
    - df: The input DataFrame to process.

    This function modifies the DataFrame in-place and does not return anything.
    """
    # List of categorical columns
    categorical_columns = ['hospital_code', 'city_code_hospital', 'bed_grade', 'city_code_patient']
    
    # Convert the specified columns to categorical (object) in-place
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    print(f"Converted the following columns to categorical: {categorical_columns}")

    # Drop unnecessary columns in-place
    df.drop(columns=['case_id', 'patientid'], inplace=True)
    print("Dropped the 'case_id' and 'patientid' columns as they are unique identifiers and not useful for modeling.")

    # Show the outcome
    print(f"Processed DataFrame shape: {df.shape}")
    print(f"Data types after processing:\n{df.dtypes}")


def target_encode(df, column, target_column=None, encoding_mapping=None):
    """
    Perform Target Encoding for a feature based on the target variable.
    
    Parameters:
    - df: DataFrame to apply the encoding on.
    - column: The feature column to apply the encoding.
    - target_column: The target column for encoding (optional).
    - encoding_mapping: Previously learned encoding mapping to apply (for test data).
    
    Returns:
    - df: The DataFrame with the encoded column.
    - encoding_mapping: The learned encoding mapping (used for applying to test data).
    """
    
    # Ensure no leading/trailing whitespace
    df[column] = df[column].astype(str).str.strip()
    
    if target_column:
        # Calculate the class probabilities for each category in the feature based on the target
        class_probs = df.groupby(column)[target_column].value_counts(normalize=True).unstack(fill_value=0)
        
        # Store the mapping of probabilities (for applying to test data)
        encoding_mapping = class_probs.to_dict(orient='index')
        
        # For each target class, map the class probabilities to the feature column
        for target_class in class_probs.columns:
            df[f'{target_column}_{target_class}_prob'] = df[column].map(lambda x: encoding_mapping.get(x, {}).get(target_class, 0))
        
        print(f"Target Encoding applied to: {column} based on target '{target_column}'")
    else:
        if encoding_mapping is None:
            raise ValueError("Encoding mapping is required for test data, but was not provided.")
        
        # Apply pre-learned encoding mapping if provided (for test data)
        for target_class in encoding_mapping[next(iter(encoding_mapping))].keys():  
            df[f'{target_column}_{target_class}_prob'] = df[column].map(lambda x: encoding_mapping.get(x, {}).get(target_class, 0))
        
        print(f"Target Encoding applied to: {column} using pre-learned mapping")
    
    # Convert the newly created probability columns to float
    prob_columns = [col for col in df.columns if 'prob' in col]
    df[prob_columns] = df[prob_columns].astype(float)
    
    return df, encoding_mapping

def encode_data(df, target_column=None, encoding_mapping=None):
    """
    Encodes the DataFrame:
    - One-Hot Encoding for nominal variables.
    - Ordinal Encoding for ordinal variables and the target if provided.
    - Frequency Encoding for categorical features, using the target column when available.
    
    Parameters:
    - df: The input DataFrame.
    - target_column: The target column (e.g., 'Stay'). Default is None.
    - encoding_mapping: A previously learned encoding mapping to apply (for test data).
    
    Returns:
    - Processed DataFrame (with encoded features and target if provided).
    - Frequency encoding mapping (for use in test data).
    """
    
    # Debug: Check the value of target_column
    print(f"Target column provided: '{target_column}'")

    # Define columns for encoding
    nominal_columns = ['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 
                       'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade']
    
    # Define ordinal columns and their mappings
    severity_mapping = {'Minor': 1, 'Moderate': 2, 'Extreme': 3}
    age_mapping = {'0-10': 1, '11-20': 2, '21-30': 3, '31-40': 4, '41-50': 5, '51-60': 6, '61-70': 7, '71-80': 8, '81-90': 9, '91-100': 10}
    admission_mapping = {'Emergency': 1, 'Trauma': 2, 'Urgent': 3}
    stay_mapping = {'0-10': 1, '11-20': 2, '21-30': 3, '31-40': 4, '41-50': 5, '51-60': 6, '61-70': 7, '71-80': 8, '81-90': 9, '91-100': 10, 'More than 100 Days': 11}

    # Ordinal Columns
    ordinal_columns = ['Severity of Illness', 'Age', 'Type of Admission', 'Stay']
    
    for col in ordinal_columns:
        if col in df.columns:
            # Convert to string type and strip leading/trailing whitespaces
            df[col] = df[col].astype(str).str.strip()

    # 1. Apply Ordinal Encoding for 'Severity of Illness', 'Age', 'Type of Admission'
    if 'Severity of Illness' in df.columns:
        df['Severity of Illness'] = df['Severity of Illness'].map(severity_mapping)
        print(f"Ordinal Encoding applied to 'Severity of Illness'")

    if 'Age' in df.columns:
        df['Age'] = df['Age'].map(age_mapping)
        print(f"Ordinal Encoding applied to 'Age'")

    if 'Type of Admission' in df.columns:
        df['Type of Admission'] = df['Type of Admission'].map(admission_mapping)
        print(f"Ordinal Encoding applied to 'Type of Admission'")

    # 2. One-Hot Encoding for nominal columns
    df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)
    print(f"One-Hot Encoding applied to: {', '.join(nominal_columns)}")

    # 3. Apply Target Encoding for 'City_Code_Patient' (only if target_column is provided)
    if 'City_Code_Patient' in df.columns:
        df, encoding_mapping = target_encode(df, column='City_Code_Patient', target_column=target_column, encoding_mapping=encoding_mapping)

    # 4. Apply Ordinal Encoding for 'Stay' after target encoding
    if 'Stay' in df.columns:
        df['Stay'] = df['Stay'].map(stay_mapping)
        print(f"Ordinal Encoding applied to 'Stay'")

    # Drop original 'City_Code_Patient' column if you want to keep only encoded columns
    if 'City_Code_Patient' in df.columns:
        df.drop(columns=['City_Code_Patient'], inplace=True)
        print(f"Dropped original column 'City_Code_Patient' after encoding")

    # Return processed DataFrame
    if target_column:
        return df, encoding_mapping  # Return features and encoding mapping if target_column is provided
    else:
        return df

    
def clean_data_inplace(df):
    """
    Cleans the DataFrame by:
    - Dropping rows with missing values (inplace).
    - Dropping duplicate rows (inplace), keeping the first occurrence.
    
    Parameters:
    - df: The input DataFrame.
    
    Returns:
    - None (the original DataFrame is modified inplace).
    """
    # Initial shape before any cleaning
    initial_shape = df.shape
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Shape after dropping NA
    shape_after_na = df.shape
    
    # Count duplicates after dropping NA
    duplicate_count_after_na = df.duplicated().sum()
    
    # If duplicates are present after dropping NA
    if duplicate_count_after_na > 0:
        # Drop duplicate rows, keeping the first occurrence
        df.drop_duplicates(keep='first', inplace=True)
        # Final shape after dropping duplicates
        final_shape = df.shape
        duplicate_count_after_drop = df.duplicated().sum()
        print(f"Initial Shape: {initial_shape}")
        print(f"Shape after dropping rows with missing values: {shape_after_na}")
        print(f"Duplicates found after dropping missing values: {duplicate_count_after_na} rows.")
        print(f"Duplicates dropped: {duplicate_count_after_na - duplicate_count_after_drop} rows removed.")
        print(f"Final Shape after cleaning (missing values and duplicates): {final_shape}")
    else:
        print(f"Initial Shape: {initial_shape}")
        print(f"Shape after dropping rows with missing values: {shape_after_na}")
        print("No duplicates found after dropping rows with missing values.")
        print(f"Final Shape after cleaning (missing values): {shape_after_na}")
        
