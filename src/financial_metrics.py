"""
Financial Metrics Module using pandas_ta and custom calculations
"""

import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("Warning: pandas_ta not available. Using manual calculations.")


def calculate_daily_returns(df, column='Close'):
    """
    Calculate daily percentage returns.
    
    Args:
        df: DataFrame with price data
        column: Column name to calculate returns on
    
    Returns:
        Series with daily returns
    """
    return df[column].pct_change()


def calculate_volatility(df, window=30, column='Close'):
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        df: DataFrame with price data
        window: Rolling window size
        column: Column name to calculate volatility on
    
    Returns:
        Series with volatility values
    """
    returns = calculate_daily_returns(df, column)
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def calculate_sharpe_ratio(df, risk_free_rate=0.02, window=252, column='Close'):
    """
    Calculate Sharpe ratio.
    
    Args:
        df: DataFrame with price data
        risk_free_rate: Annual risk-free rate (default 2%)
        window: Rolling window size
        column: Column name to calculate Sharpe ratio on
    
    Returns:
        Series with Sharpe ratio values
    """
    returns = calculate_daily_returns(df, column)
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()
    sharpe = (rolling_mean / rolling_std) * np.sqrt(252)  # Annualized
    return sharpe


def calculate_max_drawdown(df, column='Close'):
    """
    Calculate maximum drawdown.
    
    Args:
        df: DataFrame with price data
        column: Column name to calculate drawdown on
    
    Returns:
        Series with drawdown values
    """
    cumulative = (1 + calculate_daily_returns(df, column)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def add_financial_metrics(df, metrics=['returns', 'volatility']):
    """
    Add financial metrics to a DataFrame.
    
    Args:
        df: DataFrame with price data
        metrics: List of metrics to calculate
    
    Returns:
        DataFrame with added metric columns
    """
    df = df.copy()
    
    if 'returns' in metrics or 'daily_returns' in metrics:
        df['Daily_Return'] = calculate_daily_returns(df)
    
    if 'volatility' in metrics:
        df['Volatility_30d'] = calculate_volatility(df, window=30)
    
    if 'sharpe' in metrics or 'sharpe_ratio' in metrics:
        df['Sharpe_Ratio'] = calculate_sharpe_ratio(df)
    
    if 'drawdown' in metrics or 'max_drawdown' in metrics:
        df['Drawdown'] = calculate_max_drawdown(df)
    
    return df

