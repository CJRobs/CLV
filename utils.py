# utils.py
"""
Utility functions for CLV analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate input data for CLV analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input transaction data
        
    Returns:
    --------
    bool : True if data is valid, raises ValueError otherwise
    """
    required_columns = ['User_ID', 'Product_ID', 'Purchase']
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for null values in critical columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found: \n{null_counts[null_counts > 0]}")
    
    # Check for negative purchase values
    if (df['Purchase'] < 0).any():
        raise ValueError("Negative purchase values found in data")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df['Purchase']):
        raise ValueError("Purchase column must be numeric")
    
    logger.info(f"Data validation passed. Shape: {df.shape}")
    return True

def create_synthetic_dates(df: pd.DataFrame, 
                          start_date: str = '2024-01-01',
                          end_date: str = '2024-12-31') -> pd.DataFrame:
    """
    Create synthetic transaction dates for lifetime modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data
    start_date : str
        Start date for synthetic dates
    end_date : str
        End date for synthetic dates
        
    Returns:
    --------
    pd.DataFrame : Transaction data with dates
    """
    logger.info("Creating synthetic transaction dates...")
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    period_days = (end - start).days
    
    transactions_list = []
    
    for user_id in df['User_ID'].unique():
        user_data = df[df['User_ID'] == user_id]
        n_transactions = len(user_data)
        
        if n_transactions == 1:
            # Single transaction at random point
            tx_date = start + pd.Timedelta(days=np.random.randint(0, period_days))
            transactions_list.append({
                'User_ID': user_id,
                'Transaction_Date': tx_date,
                'Purchase': user_data['Purchase'].iloc[0]
            })
        else:
            # Multiple transactions spread across period
            days_between = period_days / n_transactions
            for i, (_, row) in enumerate(user_data.iterrows()):
                # Add some randomness
                day_offset = int(i * days_between + np.random.randint(-30, 30))
                day_offset = max(0, min(period_days - 1, day_offset))
                
                tx_date = start + pd.Timedelta(days=day_offset)
                transactions_list.append({
                    'User_ID': user_id,
                    'Transaction_Date': tx_date,
                    'Purchase': row['Purchase']
                })
    
    transactions_df = pd.DataFrame(transactions_list)
    logger.info(f"Created {len(transactions_df)} synthetic transactions")
    
    return transactions_df

def calculate_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RFM scores with error handling
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        DataFrame with RFM metrics
        
    Returns:
    --------
    pd.DataFrame : DataFrame with RFM scores
    """
    try:
        # Create RFM scores
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'], 
                                    q=5, 
                                    labels=[1,2,3,4,5], 
                                    duplicates='drop')
        rfm_df['M_Score'] = pd.qcut(rfm_df['Total_Purchase'], 
                                    q=5, 
                                    labels=[1,2,3,4,5], 
                                    duplicates='drop')
        
        # Handle cases where qcut fails due to too few unique values
        if rfm_df['F_Score'].isnull().any():
            logger.warning("Some F_Scores are null, using rank method")
            rfm_df['F_Score'] = pd.cut(rfm_df['Frequency'].rank(method='first'), 
                                       bins=5, 
                                       labels=[1,2,3,4,5])
        
        if rfm_df['M_Score'].isnull().any():
            logger.warning("Some M_Scores are null, using rank method")
            rfm_df['M_Score'] = pd.cut(rfm_df['Total_Purchase'].rank(method='first'), 
                                       bins=5, 
                                       labels=[1,2,3,4,5])
        
        # Combine scores
        rfm_df['RFM_Score'] = rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
        
        return rfm_df
        
    except Exception as e:
        logger.error(f"Error calculating RFM scores: {e}")
        raise

def safe_division(numerator: pd.Series, 
                  denominator: pd.Series, 
                  fill_value: float = 0) -> pd.Series:
    """
    Safely divide two series handling division by zero
    
    Parameters:
    -----------
    numerator : pd.Series
        Numerator series
    denominator : pd.Series
        Denominator series
    fill_value : float
        Value to use when denominator is zero
        
    Returns:
    --------
    pd.Series : Result of division
    """
    return np.where(denominator != 0, numerator / denominator, fill_value)

def create_summary_statistics(df: pd.DataFrame, 
                            group_by: str = 'Segment') -> pd.DataFrame:
    """
    Create summary statistics for reporting
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to summarize
    group_by : str
        Column to group by
        
    Returns:
    --------
    pd.DataFrame : Summary statistics
    """
    summary = df.groupby(group_by).agg({
        'User_ID': 'count',
        'CLV': ['sum', 'mean', 'median', 'std'],
        'Total_Purchase': 'mean',
        'Frequency': 'mean',
        'Avg_Purchase': 'mean'
    }).round(2)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary.rename(columns={'User_ID_count': 'Customer_Count'}, inplace=True)
    
    # Add percentage of total CLV
    total_clv = df['CLV'].sum()
    summary['CLV_Percentage'] = (summary['CLV_sum'] / total_clv * 100).round(2)
    
    return summary.sort_values('CLV_sum', ascending=False)

def export_results(rfm_df: pd.DataFrame, 
                  output_path: str = 'clv_results.xlsx') -> None:
    """
    Export CLV results to Excel with multiple sheets
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM data with CLV calculations
    output_path : str
        Path for output file
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main results
            rfm_df.to_excel(writer, sheet_name='CLV_Analysis', index=False)
            
            # Summary by segment
            segment_summary = create_summary_statistics(rfm_df, 'Segment')
            segment_summary.to_excel(writer, sheet_name='Segment_Summary')
            
            # Top customers
            top_100 = rfm_df.nlargest(100, 'CLV')[
                ['User_ID', 'Segment', 'CLV', 'Frequency', 'Total_Purchase']
            ]
            top_100.to_excel(writer, sheet_name='Top_100_Customers', index=False)
            
            # At-risk customers
            at_risk = rfm_df[rfm_df['Segment'].isin(['At Risk', 'About to Sleep'])]
            at_risk.to_excel(writer, sheet_name='At_Risk_Customers', index=False)
            
        logger.info(f"Results exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise