import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class CaddieNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=10):
        super(CaddieNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_caddie_model(
    data_path="data/caddie_data.csv",
    model_out_path="caddie_model.pt",
    encoders_out_path="encoders.pkl",
    label_col="recommended_club",   # the column with the "correct" or "best" club
    test_size=0.2,
    epochs=20,
    lr=0.001,
    hidden_dim=32
):
    # load the CSV dataset
    df = pd.read_csv(data_path)

    # separate features from label
    # label is the recommended club, everything else is a feature
    feature_cols = [c for c in df.columns if c != label_col]

    # label-encode the club names
    club_label_encoder = LabelEncoder()
    df[label_col] = club_label_encoder.fit_transform(df[label_col])

    # encode categorical features from inputs
    feature_encoders = {}
    X_encoded = pd.DataFrame()

    for col in feature_cols:
        if df[col].dtype == object:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(df[col].astype(str))
            feature_encoders[col] = le
        else:
            X_encoded[col] = df[col]

    # convert to numpy
    X_data = X_encoded.values.astype(np.float32)
    y_data = df[label_col].values  # integer labels

    # scale features
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data)

    # train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42, stratify=y_data
    )

    # convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test).long()

    # initialize the model
    input_dim = X_train_t.shape[1]
    output_dim = len(club_label_encoder.classes_)
    model = CaddieNet(input_dim, hidden_dim, output_dim)

    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        # track training accuracy
        predicted = torch.max(outputs, 1)
        acc = (predicted == y_train_t).float().mean()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {acc.item():.4f}")

    # evaluate on test
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, test_pred = torch.max(test_outputs, 1)
        test_acc = (test_pred == y_test_t).float().mean()
    print(f"Test Accuracy: {test_acc.item():.4f}")

    # save
    torch.save(model.state_dict(), model_out_path)
    encoder_bundle = {
        "club_label_encoder": club_label_encoder,
        "feature_encoders": feature_encoders,
        "scaler": scaler,
        "feature_cols": feature_cols
    }
    joblib.dump(encoder_bundle, encoders_out_path)

    print("Training complete!")
    print(f"Model saved to {model_out_path}")
    print(f"Encoders saved to {encoders_out_path}")

if __name__ == "__main__":
    # example usage:
    train_caddie_model()
