# old train.py

print("[DEBUG] ===== train.py is running =====")

import math
import os, json
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from model import load_parameters, save_parameters
from data import load_data
from fedn.utils.helpers.helpers import save_metadata
import traceback


TEMP_THRESHOLD = -100  # Define later based on observations

data_path = os.environ.get('FEDN_DATA_PATH')
if data_path is None:
    raise ValueError("FEDN_DATA_PATH environment variable not set!")

print(f"Training with dataset: {data_path}")


'''
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))
'''

def train(in_model_path, out_model_path):
    try:
        print(os.path.exists(in_model_path))
        print(os.path.getsize(in_model_path))
        
        data_path = os.environ.get('FEDN_DATA_PATH')
        if data_path is None:
            raise ValueError("FEDN_DATA_PATH environment variable not set!")

        print(f"Training with dataset: {data_path}")
        
        print(f"[DEBUG] Input model path (FEDn): {in_model_path}")
        print(f"[DEBUG] Output model path (FEDn expects model here): {out_model_path}")

        # --- Load and preprocess data ---
        print("[DEBUG] ===== Loading and preprocessing data...")
        X_train, y_train, recent_stats = load_data(data_path, is_train=True)
        
        print("[DEBUG] Data returned from load_data():")
        print(f" - X: {type(X_train)}, y: {type(y_train)}")
        print(f" - X shape: {getattr(X_train, 'shape', 'no shape')}")
        print(f" - y shape: {getattr(y_train, 'shape', 'no shape')}")

        if X_train is None or y_train is None:
            raise ValueError("❌ X or y is None! Likely issue inside load_data()")
        
        
        chunk_size = 80000  # You can experiment with this number!
        # Read only the first chunk
        chunk_iter = pd.read_csv(data_path, chunksize=chunk_size)
        df = next(chunk_iter)  # Get the first chunk

        # Example of datetime creation, with error handling
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True, errors='coerce')
        df = df.sort_values('timestamp')
        
        # --- Check if training should proceed (14-day mean temp check) ---
        #df = pd.read_csv(data_path)
        #df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True, errors='coerce')
        last_14_days = df.sort_values("timestamp", ascending=False).head(14)
        mean_temp = last_14_days["batMaxTemp"].mean()

        print(f"[DEBUG] Mean battery max temperature for last 14 days: {mean_temp}")

        if mean_temp <= TEMP_THRESHOLD:
            print(f"[INFO] Temperature threshold not exceeded ({mean_temp:.2f}°C). Skipping training.")
            save_metadata({}, out_model_path)  # Empty update to avoid None errors in FEDn
            return

        # --- Convert data to PyTorch tensors ---
        print("[DEBUG] Converting data to tensors...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        # --- Load model ---
        print("[DEBUG] ===== Loading model parameters...")
        model = load_parameters(in_model_path)
        

        # --- Training setup ---
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        lr = 0.01
        epochs = 2
        batch_size = 32
        n_batches = int(math.ceil(len(X_train_tensor) / batch_size))

        print(f"[DEBUG] Starting training: epochs={epochs}, batch_size={batch_size}, total_batches={n_batches}")

        # --- Training loop ---
        for epoch in range(epochs):
            for b in range(n_batches):
                batch_x = X_train_tensor[b * batch_size : (b + 1) * batch_size]
                batch_y = y_train_tensor[b * batch_size : (b + 1) * batch_size]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                print(f"[DEBUG] Epoch [{epoch}/{epochs}] Loss: {loss.item():.4f}")

        # --- MANDATORY for FEDn ---
        print("[DEBUG] Saving metadata and model parameters...")
        metadata = {
            "num_examples": len(X_train_tensor),
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,  # or 0.001 if fixed
            "temperature_avg": recent_stats["temperature_avg"],
            "batMaxTemp_avg": recent_stats["batMaxTemp_avg"],
            "socPercent_avg": recent_stats["socPercent_avg"],
            "chargeCycle_std": recent_stats["chargeCycle_std"]
        }


        save_metadata(metadata, out_model_path)
        save_parameters(model, out_model_path)
        print(f"[DEBUG] ===== Model saved to: {out_model_path}, Exists: {os.path.exists(out_model_path)}")
        
        if not os.path.exists(out_model_path):
            raise FileNotFoundError(f"❌ Model file was not saved to: {out_model_path}")


    except Exception as e:
        print("\n\n[ERROR] Training failed with exception:\n")
        traceback.print_exc()
        # Also write a metadata to avoid FEDn crashing on NoneType returns
        save_metadata({}, out_model_path)


# === Entry point ===
if __name__ == "__main__":
    print("\n\n[DEBUG] ===== Starting train() via __main__ =====\n\n")
    if len(sys.argv) < 2:
        print("[ERROR] Not enough arguments provided to train.py")
        sys.exit(1)
    try:
        train(sys.argv[1], sys.argv[2]) #used have a datapath here as a third parameters in the train() function but removed it
    except Exception as e:
        print("\n\n[ERROR] Something went wrong in main train.py execution!\n")
        traceback.print_exc()

