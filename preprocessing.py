import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import config

def preprocess():
    print("Loading datasets...")

    normal = pd.read_csv(os.path.join(config.DATA_PATH, "normal_traffic_fixed.csv"))
    dos = pd.read_csv(os.path.join(config.DATA_PATH, "dos_traffic_fixed.csv"))
    probe = pd.read_csv(os.path.join(config.DATA_PATH, "probe_traffic_fixed.csv"))

    # Assign labels
    normal["Label"] = 0
    dos["Label"] = 1
    probe["Label"] = 2

    print("Merging datasets...")

    df = pd.concat([normal, dos, probe], ignore_index=False)

    print("Dataset size:", df.shape)

    df["Label"] = df["Label"].astype(int)

    # Remove unwanted columns
    X = df.drop(columns=config.EXCLUDE_COLS, errors="ignore")
    y = df["Label"]

    # Save feature names
    feature_names = X.columns.tolist()

    print("Number of features:", X.shape[1])

    # Replace infinite values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN
    X.fillna(0, inplace=True)

    # Save original values
    X_original = X.values.copy()

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    print("Tensor shapes:")
    print("Features:", X_tensor.shape)
    print("Labels:", y_tensor.shape)

    torch.save(
        {
            "features": X_tensor,
            "labels": y_tensor,
            "features_original": X_original,
            "input_dim": X_tensor.shape[1],
            "feature_names": feature_names,
            "scaler": scaler
        },
        config.PROCESSED_FILE,
    )

    print("Processed dataset saved at:", config.PROCESSED_FILE)


if __name__ == "__main__":
    preprocess()