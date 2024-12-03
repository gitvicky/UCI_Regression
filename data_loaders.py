# %% 
import pandas as pd 
import numpy as np 
import torch
import os 
from ucimlrepo import fetch_ucirepo 

#Loading Data 
data_loc = os.getcwd() + '/Data'

# %%

import pandas as pd
import numpy as np

def analyze_and_clean_nans(df, threshold=0):
    """
    Analyze NaN values in a DataFrame and remove columns based on NaN threshold.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame to analyze
    threshold (float): If a column has more than this percentage of NaNs, it will be removed
                      0 means remove columns with any NaNs
                      Default is 0
    
    Returns:
    tuple: (cleaned_df, summary_dict)
        - cleaned_df: DataFrame with NaN columns removed
        - summary_dict: Dictionary containing NaN analysis results
    """
    # Get count of NaNs in each column
    nan_counts = df.isna().sum()
    
    # Get percentage of NaNs in each column
    nan_percentages = (df.isna().sum() / len(df) * 100).round(2)
    
    # Identify columns with NaNs
    columns_with_nans = nan_counts[nan_counts > 0].index.tolist()
    
    # Identify columns to remove (those exceeding threshold)
    columns_to_remove = df.columns[nan_percentages > threshold].tolist()
    
    # Create cleaned DataFrame
    cleaned_df = df.drop(columns=columns_to_remove)
    
    # Create summary statistics
    summary = {
        'original_column_count': len(df.columns),
        'columns_removed': columns_to_remove,
        'remaining_column_count': len(cleaned_df.columns),
        'columns_with_nans': columns_with_nans,
        'total_nan_count': nan_counts.sum(),
        'nan_counts': nan_counts[columns_with_nans].to_dict(),
        'nan_percentages': nan_percentages[columns_with_nans].to_dict(),
        'columns_all_nan': df.columns[df.isna().all()].tolist(),
        'rows_with_any_nan': df.isna().any(axis=1).sum()
    }
    
    return cleaned_df, summary

import pandas as pd

def filter_matching_nans(inputs_df, targets_df):
    """
    Filter out rows containing NaN values from both input and target dataframes,
    maintaining the alignment between them.
    
    Parameters:
    -----------
    inputs_df : pandas.DataFrame
        DataFrame containing model input features
    targets_df : pandas.DataFrame
        DataFrame containing target values
        
    Returns:
    --------
    tuple
        (filtered_inputs, filtered_targets): Pair of DataFrames with NaN rows removed
    """
    # Ensure dataframes have the same number of rows
    if len(inputs_df) != len(targets_df):
        raise ValueError("Input and target dataframes must have the same number of rows")
    
    # Create masks for non-NaN values in both dataframes
    inputs_mask = ~inputs_df.isna().any(axis=1)
    targets_mask = ~targets_df.isna().any(axis=1)
    
    # Combine masks to get rows that have no NaNs in either dataframe
    combined_mask = inputs_mask & targets_mask
    
    # Apply the mask to both dataframes
    filtered_inputs = inputs_df[combined_mask].reset_index(drop=True)
    filtered_targets = targets_df[combined_mask].reset_index(drop=True)
    
    return filtered_inputs, filtered_targets


def df_to_pytorch_tensor(df, numeric_only=False):
    """
    Convert DataFrame to PyTorch tensor.
    
    Args:
        df: pandas DataFrame
        numeric_only: if True, only convert numeric columns
    
    Returns:
        PyTorch tensor
    """
    if numeric_only:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        return torch.tensor(numeric_df.values, dtype=torch.float32)
    else:
        # Convert categorical columns to numeric first
        df_encoded = pd.get_dummies(df)
        return torch.tensor(df_encoded.values, dtype=torch.float32)


def sgemm():
    df = pd.read_csv(data_loc + '/sgemm_product.csv')
    inputs = df.iloc[:, 0:14].to_numpy()
    outputs = df.iloc[:, 14:].to_numpy()

    inputs, outputs = torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

    case_config = {"Case": 'sgemm',
                   "in_dims": 14, 
                   "out_dims": 4
                    }
    return case_config, inputs, outputs

def crimes_and_community():
    # fetch dataset 
    communities_and_crime_unnormalized = fetch_ucirepo(id=211) 
    
    # data (as pandas dataframes) 
    X = communities_and_crime_unnormalized.data.features 
    y = communities_and_crime_unnormalized.data.targets 
  
    X, _ = analyze_and_clean_nans(X)
    X['State'] = np.arange(len(X)) #One-hot encoding. 

    X, y = filter_matching_nans(X, y)

    X = df_to_pytorch_tensor(X)
    y = df_to_pytorch_tensor(y)

    case_config = {"Case": 'crimes_and_community',
                "in_dims": X.shape[1], 
                "out_dims": y.shape[1]
                }
    
    return case_config, X, y

def bias_correction():
    df = pd.read_csv(data_loc + '/Bias_correction_ucl.csv')
    df, _ = filter_matching_nans(df, df)
    df = df.drop(columns='Date')

    inputs = df.iloc[:, 0:22].to_numpy()
    outputs = df.iloc[:, 22:].to_numpy()

    inputs, outputs = torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)
    
    case_config = {"Case": 'bias_correction',
                   "in_dims": inputs.shape[1], 
                   "out_dims": outputs.shape[1]
                    }
    
    return case_config, inputs, outputs


def music_origin():
    df = pd.read_csv(data_loc + '/default_features_1059_tracks.txt', header=None)

    inputs = df.iloc[:, 0:68].to_numpy()
    outputs = df.iloc[:, 68:].to_numpy()

    inputs, outputs = torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

        
    case_config = {"Case": 'bias_correction',
                   "in_dims": inputs.shape[1], 
                   "out_dims": outputs.shape[1]
                    }
    
    return case_config, inputs, outputs

def indoor_localisation():
    df1 = pd.read_csv(data_loc + '/indoorloc_trainingData.csv')
    df2 = pd.read_csv(data_loc + '/indoorloc_validationData.csv')

    df = pd.concat(
            [df1, df2],
            axis=0,  # 0 for row-wise concatenation
            ignore_index=True
        )
    

    inputs = df.iloc[:, 0:520].to_numpy()
    outputs = df.iloc[:, 520:522].to_numpy()

    inputs, outputs = torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

        
    case_config = {"Case": 'bias_correction',
                   "in_dims": inputs.shape[1], 
                   "out_dims": outputs.shape[1]
                    }
    
    return case_config, inputs, outputs

def load_data(case):

    if case == 'sgemm':
        return sgemm()
    if case == 'crimes_and_community':
        return crimes_and_community()
    if case == 'bias_correction':
        return bias_correction()
    if case == 'music_origin':
        return music_origin()
    if case == 'indoor_localisation':
        return indoor_localisation()
    
# %% 

  
