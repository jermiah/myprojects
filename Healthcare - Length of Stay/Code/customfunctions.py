import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def summarize_nulls_and_modalities(df, columns):
    """
    This function summarizes missing data and unique values for specified columns, returning null counts, percentages, unique values (modalities), and data types in a sorted DataFrame.
    """
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
 
    subset_df = df.iloc[:, :2]  # This selects all rows and the first two columns

    # Filter the subset DataFrame where the first column matches the 'variables' list
    result_df = subset_df[subset_df.iloc[:, 0].isin(variables)]

    return result_df.reset_index(drop=True)


def plot_categorical_variables(data, categorical_variables, orientation='vertical'):

    num_vars = len(categorical_variables)
    cols = math.ceil(math.sqrt(num_vars))  # Calculate number of columns
    rows = math.ceil(num_vars / cols)     # Calculate number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    if num_vars == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()

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
        # max_value = data[variable].max()
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # Add 15% padding to y-axis

    # Remove unused axes
    if num_vars < len(axes):
        for j in range(num_vars, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()  # Automatically adjust subplot spacing
    plt.show()
  

def create_pairplot(df, corner=True):
    
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
        
