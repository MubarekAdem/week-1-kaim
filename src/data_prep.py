import pandas as pd
import os


def load_news(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['headline'] = df['headline'].astype(str)
    df['publisher'] = df['publisher'].astype(str)
    return df


def load_stock_data(path, symbol=None):
    """
    Load stock price data from CSV file.
    
    Args:
        path: Path to the CSV file or directory containing stock CSV files
        symbol: Stock symbol (e.g., 'AAPL'). If None and path is a file, 
                symbol is inferred from filename.
    
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns
    """
    if os.path.isdir(path):
        if symbol is None:
            raise ValueError("symbol must be provided when path is a directory")
        file_path = os.path.join(path, f"{symbol}.csv")
    else:
        file_path = path
        if symbol is None:
            # Infer symbol from filename
            symbol = os.path.basename(file_path).replace('.csv', '')
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Symbol'] = symbol
    return df


def load_multiple_stocks(data_dir, symbols=None):
    """
    Load multiple stock data files.
    
    Args:
        data_dir: Directory containing stock CSV files
        symbols: List of stock symbols to load. If None, loads all CSV files.
    
    Returns:
        Dictionary of DataFrames keyed by symbol
    """
    if symbols is None:
        # Find all CSV files in directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        symbols = [f.replace('.csv', '') for f in csv_files if f != 'raw_analyst_ratings.csv']
    
    stocks = {}
    for symbol in symbols:
        try:
            stocks[symbol] = load_stock_data(data_dir, symbol)
        except FileNotFoundError:
            print(f"Warning: Could not load data for {symbol}")
    
    return stocks
