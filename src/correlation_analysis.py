"""
Correlation Analysis Module
Analyzes correlation between news sentiment and stock price movements
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Import calculate_daily_returns - handle both relative and absolute imports
try:
    from .financial_metrics import calculate_daily_returns
except ImportError:
    from src.financial_metrics import calculate_daily_returns


def align_dates(news_df, stock_df, news_date_col='date', stock_date_col='Date'):
    """
    Align news and stock data by dates.

    Args:
        news_df: DataFrame with news data (must have date column and stock column)
        stock_df: DataFrame with stock data (must have date column)
        news_date_col: Name of date column in news_df
        stock_date_col: Name of date column in stock_df

    Returns:
        Merged DataFrame with aligned dates
    """
    # Ensure dates are datetime
    news_df = news_df.copy()
    stock_df = stock_df.copy()

    news_df[news_date_col] = pd.to_datetime(news_df[news_date_col])
    stock_df[stock_date_col] = pd.to_datetime(stock_df[stock_date_col])

    # Extract date only (remove time component) for alignment
    news_df['date_only'] = news_df[news_date_col].dt.date
    stock_df['date_only'] = stock_df[stock_date_col].dt.date

    # Merge on date
    merged = pd.merge(
        news_df,
        stock_df,
        left_on='date_only',
        right_on='date_only',
        how='inner'
    )

    return merged


def calculate_daily_sentiment(news_df, date_col='date', sentiment_col='sentiment'):
    """
    Calculate daily average sentiment scores.

    Args:
        news_df: DataFrame with news data
        date_col: Name of date column
        sentiment_col: Name of sentiment column

    Returns:
        DataFrame with date and average sentiment
    """
    news_df = news_df.copy()
    news_df[date_col] = pd.to_datetime(news_df[date_col])
    news_df['date_only'] = news_df[date_col].dt.date

    daily_sentiment = news_df.groupby('date_only')[sentiment_col].agg([
        'mean',
        'std',
        'count'
    ]).reset_index()
    daily_sentiment.columns = [
        'Date', 'Avg_Sentiment', 'Sentiment_Std', 'News_Count']

    return daily_sentiment


def merge_sentiment_and_returns(news_df, stock_df,
                                news_date_col='date',
                                stock_date_col='Date',
                                sentiment_col='sentiment',
                                stock_col=None):
    """
    Merge daily sentiment scores with stock returns.

    Args:
        news_df: DataFrame with news data
        stock_df: DataFrame with stock data
        news_date_col: Name of date column in news_df
        stock_date_col: Name of date column in stock_df
        sentiment_col: Name of sentiment column
        stock_col: Name of stock symbol column in news_df

    Returns:
        DataFrame with aligned sentiment and returns
    """
    # Calculate daily sentiment
    daily_sentiment = calculate_daily_sentiment(
        news_df, news_date_col, sentiment_col)

    # Calculate daily returns
    stock_df = stock_df.copy()
    stock_df['Daily_Return'] = calculate_daily_returns(stock_df)

    # Prepare stock data
    stock_df[stock_date_col] = pd.to_datetime(stock_df[stock_date_col])
    stock_df['date_only'] = stock_df[stock_date_col].dt.date

    # Merge
    merged = pd.merge(
        daily_sentiment,
        stock_df[[stock_date_col, 'date_only', 'Daily_Return', 'Close']],
        left_on='Date',
        right_on='date_only',
        how='inner'
    )

    return merged


def calculate_correlation(merged_df, sentiment_col='Avg_Sentiment', return_col='Daily_Return'):
    """
    Calculate correlation between sentiment and returns.

    Args:
        merged_df: DataFrame with sentiment and returns
        sentiment_col: Name of sentiment column
        return_col: Name of returns column

    Returns:
        Dictionary with correlation metrics
    """
    # Remove NaN values
    clean_df = merged_df[[sentiment_col, return_col]].dropna()

    if len(clean_df) < 2:
        return {
            'pearson_correlation': np.nan,
            'pearson_pvalue': np.nan,
            'spearman_correlation': np.nan,
            'spearman_pvalue': np.nan,
            'n_observations': len(clean_df)
        }

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(
        clean_df[sentiment_col], clean_df[return_col])

    # Spearman correlation
    spearman_r, spearman_p = spearmanr(
        clean_df[sentiment_col], clean_df[return_col])

    return {
        'pearson_correlation': pearson_r,
        'pearson_pvalue': pearson_p,
        'spearman_correlation': spearman_r,
        'spearman_pvalue': spearman_p,
        'n_observations': len(clean_df)
    }


def analyze_sentiment_impact(news_df, stock_df,
                             news_date_col='date',
                             stock_date_col='Date',
                             sentiment_col='sentiment',
                             stock_col='stock'):
    """
    Comprehensive analysis of sentiment impact on stock returns.

    Args:
        news_df: DataFrame with news data
        stock_df: DataFrame with stock data
        news_date_col: Name of date column in news_df
        stock_date_col: Name of date column in stock_df
        sentiment_col: Name of sentiment column
        stock_col: Name of stock symbol column in news_df

    Returns:
        Dictionary with analysis results
    """
    # Merge sentiment and returns
    merged = merge_sentiment_and_returns(
        news_df, stock_df, news_date_col, stock_date_col, sentiment_col, stock_col
    )

    # Calculate correlation
    correlation = calculate_correlation(merged)

    # Additional statistics
    positive_sentiment_days = merged[merged['Avg_Sentiment'] > 0]
    negative_sentiment_days = merged[merged['Avg_Sentiment'] < 0]

    results = {
        'correlation_metrics': correlation,
        'total_days': len(merged),
        'positive_sentiment_days': len(positive_sentiment_days),
        'negative_sentiment_days': len(negative_sentiment_days),
        'avg_return_positive_sentiment': positive_sentiment_days['Daily_Return'].mean() if len(positive_sentiment_days) > 0 else np.nan,
        'avg_return_negative_sentiment': negative_sentiment_days['Daily_Return'].mean() if len(negative_sentiment_days) > 0 else np.nan,
        'merged_data': merged
    }

    return results
