"""
Technical Indicators Module
Calculates technical indicators using TA-Lib and pandas_ta
"""

import pandas as pd

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using pandas_ta as fallback.")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("Warning: pandas_ta not available.")


def calculate_sma(df, period=50, column='Close'):
    """
    Calculate Simple Moving Average.

    Args:
        df: DataFrame with price data
        period: Period for moving average
        column: Column name to calculate MA on

    Returns:
        Series with SMA values
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.SMA(df[column].values, timeperiod=period), index=df.index)
    elif PANDAS_TA_AVAILABLE:
        return ta.sma(df[column], length=period)
    else:
        return df[column].rolling(window=period).mean()


def calculate_ema(df, period=12, column='Close'):
    """
    Calculate Exponential Moving Average.

    Args:
        df: DataFrame with price data
        period: Period for EMA
        column: Column name to calculate EMA on

    Returns:
        Series with EMA values
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.EMA(df[column].values, timeperiod=period), index=df.index)
    elif PANDAS_TA_AVAILABLE:
        return ta.ema(df[column], length=period)
    else:
        return df[column].ewm(span=period, adjust=False).mean()


def calculate_rsi(df, period=14, column='Close'):
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: DataFrame with price data
        period: Period for RSI calculation
        column: Column name to calculate RSI on

    Returns:
        Series with RSI values (0-100)
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.RSI(df[column].values, timeperiod=period), index=df.index)
    elif PANDAS_TA_AVAILABLE:
        return ta.rsi(df[column], length=period)
    else:
        # Manual RSI calculation
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def calculate_macd(df, fastperiod=12, slowperiod=26, signalperiod=9, column='Close'):
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        df: DataFrame with price data
        fastperiod: Fast EMA period
        slowperiod: Slow EMA period
        signalperiod: Signal line period
        column: Column name to calculate MACD on

    Returns:
        DataFrame with MACD, MACD_signal, and MACD_hist columns
    """
    if TALIB_AVAILABLE:
        macd, signal, hist = talib.MACD(
            df[column].values,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        return pd.DataFrame({
            'MACD': macd,
            'MACD_signal': signal,
            'MACD_hist': hist
        }, index=df.index)
    elif PANDAS_TA_AVAILABLE:
        macd_df = ta.macd(df[column], fast=fastperiod,
                          slow=slowperiod, signal=signalperiod)
        return macd_df
    else:
        # Manual MACD calculation
        ema_fast = df[column].ewm(span=fastperiod, adjust=False).mean()
        ema_slow = df[column].ewm(span=slowperiod, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        hist = macd - signal
        return pd.DataFrame({
            'MACD': macd,
            'MACD_signal': signal,
            'MACD_hist': hist
        }, index=df.index)


def calculate_bollinger_bands(df, period=20, std_dev=2, column='Close'):
    """
    Calculate Bollinger Bands.

    Args:
        df: DataFrame with price data
        period: Period for moving average
        std_dev: Number of standard deviations
        column: Column name to calculate bands on

    Returns:
        DataFrame with upper, middle, and lower bands
    """
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(
            df[column].values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev
        )
        return pd.DataFrame({
            'BB_upper': upper,
            'BB_middle': middle,
            'BB_lower': lower
        }, index=df.index)
    elif PANDAS_TA_AVAILABLE:
        bb = ta.bbands(df[column], length=period, std=std_dev)
        return bb
    else:
        # Manual calculation
        middle = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return pd.DataFrame({
            'BB_upper': upper,
            'BB_middle': middle,
            'BB_lower': lower
        }, index=df.index)


def add_technical_indicators(df, indicators=None):
    """
    Add multiple technical indicators to a DataFrame.

    Args:
        df: DataFrame with price data (must have Open, High, Low, Close, Volume)
        indicators: List of indicators to calculate

    Returns:
        DataFrame with added indicator columns
    """
    if indicators is None:
        indicators = ['SMA_50', 'RSI', 'MACD']
    df = df.copy()

    if 'SMA_50' in indicators:
        df['SMA_50'] = calculate_sma(df, period=50)

    if 'SMA_200' in indicators:
        df['SMA_200'] = calculate_sma(df, period=200)

    if 'EMA_12' in indicators:
        df['EMA_12'] = calculate_ema(df, period=12)

    if 'EMA_26' in indicators:
        df['EMA_26'] = calculate_ema(df, period=26)

    if 'RSI' in indicators:
        df['RSI'] = calculate_rsi(df, period=14)

    if 'MACD' in indicators:
        macd_df = calculate_macd(df)
        df['MACD'] = macd_df['MACD']
        df['MACD_signal'] = macd_df['MACD_signal']
        df['MACD_hist'] = macd_df['MACD_hist']

    if 'BB' in indicators or 'Bollinger' in indicators:
        bb_df = calculate_bollinger_bands(df)
        df['BB_upper'] = bb_df['BB_upper']
        df['BB_middle'] = bb_df['BB_middle']
        df['BB_lower'] = bb_df['BB_lower']

    return df
