import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_nse_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        data[ticker] = stock_data
    return data

def aggregate_weekly_data(data):
    aggregated_data = {}
    for ticker, df in data.items():
        df['Week'] = df.index.to_period('W').astype(str)
        weekly_data = df.groupby('Week').agg({'Open': 'first', 'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'})
        aggregated_data[ticker] = weekly_data
    return aggregated_data

def save_to_csv(aggregated_data, filename):
    combined_df = pd.concat(aggregated_data.values(), keys=aggregated_data.keys())
    combined_df = combined_df.reset_index(level=0).rename(columns={'level_0': 'Ticker'})
    combined_df.to_csv(filename)

def main():
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']  # Example tickers
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    nse_data = fetch_nse_data(tickers, start_date, end_date)
    aggregated_data = aggregate_weekly_data(nse_data)
    save_to_csv(aggregated_data, 'nse_stock_data.csv')

if __name__ == "__main__":
    main()