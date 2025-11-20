"""
Data Loading Module
Handles downloading and loading the Online Retail II dataset from Kaggle
"""

import kagglehub
import pandas as pd


def load_retail_data():
    """
    Download and load the Online Retail II dataset from Kaggle
    
    Returns:
        pd.DataFrame: Raw dataset with all transactions
    """
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("mashlyn/online-retail-ii-uci")
    file_path = path + "/online_retail_II.csv"
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    
    print(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    return df


def get_data_summary(df):
    """
    Generate a summary of the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': df.columns.tolist(),
        'missing_values': df.isna().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    return summary


if __name__ == "__main__":
    # Test the module
    df = load_retail_data()
    print("\nDataset Preview:")
    print(df.head())
    print("\nData Summary:")
    summary = get_data_summary(df)
    print(f"Total Rows: {summary['total_rows']}")
    print(f"Total Columns: {summary['total_columns']}")
    print(f"Columns: {summary['columns']}")
