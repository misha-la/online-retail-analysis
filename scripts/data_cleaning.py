"""
Data Cleaning Module
Handles data cleaning, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np


def clean_retail_data(df):
    """
    Clean and preprocess the retail dataset
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (df_sales, df_returns) - cleaned sales data and returns data
    """
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Separate returns/cancellations from sales
    df_returns = df[df['Invoice'].astype(str).str.startswith('C')].copy()
    df_sales = df[~df['Invoice'].astype(str).str.startswith('C')].copy()
    
    # Remove invalid transactions
    df_sales = df_sales[df_sales['Quantity'] > 0]
    df_sales = df_sales[df_sales['Price'] > 0]
    
    # Drop rows without customer ID
    df_sales = df_sales.dropna(subset=['Customer ID'])
    
    print(f"Original dataset: {len(df)} rows")
    print(f"Sales transactions: {len(df_sales)} rows")
    print(f"Returns/cancellations: {len(df_returns)} rows")
    
    return df_sales, df_returns


def engineer_features(df_sales):
    """
    Create additional features for analysis
    
    Args:
        df_sales (pd.DataFrame): Cleaned sales data
        
    Returns:
        pd.DataFrame: Sales data with engineered features
    """
    # Create TotalPrice column
    df_sales['TotalPrice'] = df_sales['Quantity'] * df_sales['Price']
    
    # Extract time-based features
    df_sales['Year'] = df_sales['InvoiceDate'].dt.year
    df_sales['Month'] = df_sales['InvoiceDate'].dt.month
    df_sales['YearMonth'] = df_sales['InvoiceDate'].dt.to_period('M').astype(str)
    df_sales['DayOfWeek'] = df_sales['InvoiceDate'].dt.day_name()
    df_sales['Hour'] = df_sales['InvoiceDate'].dt.hour
    
    print("Features engineered successfully")
    print(f"New columns: TotalPrice, Year, Month, YearMonth, DayOfWeek, Hour")
    
    return df_sales


def check_data_quality(df):
    """
    Perform data quality checks
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Data quality metrics
    """
    quality_metrics = {
        'duplicates': df.duplicated().sum(),
        'empty_rows': df.isnull().all(axis=1).sum(),
        'missing_by_column': df.isna().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    print(f"\nData Quality Report:")
    print(f"Duplicate rows: {quality_metrics['duplicates']}")
    print(f"Empty rows: {quality_metrics['empty_rows']}")
    
    return quality_metrics


if __name__ == "__main__":
    from data_loader import load_retail_data
    
    # Test the module
    df = load_retail_data()
    df_sales, df_returns = clean_retail_data(df)
    df_sales = engineer_features(df_sales)
    quality = check_data_quality(df_sales)
    
    print("\nCleaned data preview:")
    print(df_sales.head())
