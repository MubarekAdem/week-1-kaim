import pandas as pd


def load_news(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['headline'] = df['headline'].astype(str)
    df['publisher'] = df['publisher'].astype(str)
    return df
