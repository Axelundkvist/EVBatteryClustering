#!/usr/bin/env python3
print("[DEBUG] ===== validate.py is running =====")

import os
import sys
import torch
import traceback
import pandas as pd
import numpy as np

from model import load_parameters
from data import load_data
from fedn.utils.helpers.helpers import load_metadata, save_metrics

# Configuration — mirror train.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FEDn data path
data_path = os.environ.get('FEDN_DATA_PATH')
if data_path is None:
    raise ValueError("FEDN_DATA_PATH environment variable not set!")

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same preprocessing steps as in train.py."""
    df['fast_charge']        = df['fast_charge'].fillna(0)
    df['depth_of_discharge'] = df['depth_of_discharge'].fillna(0)
    df['session_type_charge']    = df['session_type_charge'].astype(float)
    df['session_type_discharge'] = df['session_type_discharge'].astype(float)
    return df.astype('float32')

def validate(in_model_path: str, out_json_path: str):
    # --- Load metadata (optional) ---
    try:
        meta = load_metadata(in_model_path)
        lr = meta.get("learning_rate", None)
        current_round = meta.get("current_round", None)
        print(f"[DEBUG] Loaded metadata: round={current_round}, lr={lr}")
    except Exception:
        print(f"[WARN] No metadata found at {in_model_path}. Skipping metadata load.")

    # --- Load and preprocess data ---
    print("[DEBUG] Loading and preprocessing training data...")
    X_train, y_train, recent_stats = load_data(data_path, is_train=True)
    X_train = preprocess_df(X_train)
    print(f"[DEBUG]  • Train set: X={X_train.shape}, y={y_train.shape}")

    print("[DEBUG] Loading and preprocessing test data...")
    X_test, y_test, _ = load_data(data_path, is_train=False)
    X_test = preprocess_df(X_test)
    print(f"[DEBUG]  • Test set:  X={X_test.shape}, y={y_test.shape}")

    # --- Convert to tensors & move to device ---
    x_train = torch.from_numpy(X_train.values).to(device)
    y_train = torch.from_numpy(y_train.values).float().view(-1, 1).to(device)
    x_test  = torch.from_numpy(X_test.values).to(device)
    y_test  = torch.from_numpy(y_test.values).float().view(-1, 1).to(device)

    print(f"[DEBUG] Data moved to device: {device}")

    # --- Load model with correct input dimension ---
    num_features = X_train.shape[1]
    print(f"[DEBUG] Instantiating model with {num_features} input features")
    model = load_parameters(in_model_path, num_features)
    model.to(device)
    model.eval()
    print("[DEBUG] Model loaded and set to eval mode.")

    # --- Compute losses ---
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        train_pred = model(x_train)
        train_loss = criterion(train_pred, y_train).item()
        print(f"[DEBUG] Training Loss: {train_loss:.6f}")

        test_pred = model(x_test)
        test_loss = criterion(test_pred, y_test).item()
        print(f"[DEBUG] Test Loss:     {test_loss:.6f}")

    # --- Save metrics back to FEDn ---
    recent_stats_serializable = recent_stats.to_dict(orient='list')

    metrics = {
        "training_loss": train_loss,
        "test_loss":     test_loss,
        **({"recent_stats": recent_stats_serializable} if recent_stats_serializable is not None else {})
    }
    save_metrics(metrics, out_json_path)
    print(f"[INFO] Validation results saved to {out_json_path}")

if __name__ == "__main__":
    print("\n\n[DEBUG] ===== Starting validate() via __main__ =====\n")
    if len(sys.argv) < 3:
        print("[ERROR] Not enough arguments provided.")
        print("Usage: python validate.py <in_model_path> <out_json_path>")
        sys.exit(1)
    try:
        validate(sys.argv[1], sys.argv[2])
    except Exception:
        print("\n\n[ERROR] Something went wrong in validate.py execution!\n")
        traceback.print_exc()
        sys.exit(1)
