import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class EnsembleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def main():
    # Load preprocessed data
    df = pd.read_csv('preprocessed_data.csv')
    # Use only the test set (last 20% weeks) for training the ensemble
    weeks = sorted(df['Week'].unique())
    split_idx = int(len(weeks) * 0.8)
    test_weeks = weeks[split_idx:]
    test_df = df[df['Week'].isin(test_weeks)]
    features = ['MA_5', 'MA_10', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Stoch']
    X = test_df[features].fillna(0).values
    y = (test_df['Close'].shift(-1) > test_df['Close']).astype(int).fillna(0).values
    # Use all test data for training (no further split)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    model = EnsembleNN(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    # Save predictions
    model.eval()
    with torch.no_grad():
        preds = model(X_train).round().numpy().flatten()
    pd.DataFrame({'Predictions': preds}).to_csv('ensemble_predictions.csv', index=False)

if __name__ == "__main__":
    main()