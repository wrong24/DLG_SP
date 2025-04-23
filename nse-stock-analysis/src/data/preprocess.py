import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def add_technical_indicators(data):
    data = data.sort_values(['Ticker', 'Week'])
    for ticker in data['Ticker'].unique():
        mask = data['Ticker'] == ticker
        data.loc[mask, 'RSI'] = ta.momentum.RSIIndicator(data.loc[mask, 'Close'], window=14).rsi().values
        macd = ta.trend.MACD(data.loc[mask, 'Close'])
        data.loc[mask, 'MACD'] = macd.macd().values
        bb = ta.volatility.BollingerBands(data.loc[mask, 'Close'])
        data.loc[mask, 'BB_High'] = bb.bollinger_hband().values
        data.loc[mask, 'BB_Low'] = bb.bollinger_lband().values
        stoch = ta.momentum.StochasticOscillator(data.loc[mask, 'High'], data.loc[mask, 'Low'], data.loc[mask, 'Close'])
        data.loc[mask, 'Stoch'] = stoch.stoch().values
    return data

def walk_forward_split(data, window=12):
    weeks = sorted(data['Week'].unique())
    split_idx = int(len(weeks) * 0.8)
    train_weeks = weeks[:split_idx]
    test_weeks = weeks[split_idx:]
    for i in range(len(test_weeks) - window + 1):
        train_window = train_weeks + test_weeks[:i]
        test_window = test_weeks[i:i+window]
        train_df = data[data['Week'].isin(train_window)]
        test_df = data[data['Week'].isin(test_window)]
        yield train_df, test_df

def preprocess_data(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    data = add_technical_indicators(data)
    data['MA_5'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).mean())
    data['MA_10'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Stoch']
    for week in data['Week'].unique():
        mask = data['Week'] == week
        scaler = MinMaxScaler()
        data.loc[mask, features] = scaler.fit_transform(data.loc[mask, features].fillna(0))
    # No change to splitting here; test set will be used in model scripts
    data.to_csv(output_csv, index=False)

if __name__ == "__main__":
    preprocess_data('nse_stock_data.csv', 'preprocessed_data.csv')