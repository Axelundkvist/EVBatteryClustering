print("[DEBUG] ===== newTrain.py is running =====")

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
# don't forget to update the model class name and the file to your newModel.py
from client.model import BatterySoHModel, load_parameters, save_parameters
from fedn.utils.helpers.helpers import load_metadata, save_metadata
from newData import load_data
import traceback

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 2

def train(in_model_path, out_model_path):
    try:
        print(f"[DEBUG] Input model path: {in_model_path}")
        print(f"[DEBUG] Output model path: {out_model_path}")

        # --- Load and preprocess data ---
        print("[DEBUG] Loading and preprocessing data...")
        X_train, y_train, recent_stats = load_data(os.environ.get('FEDN_DATA_PATH'), is_train=True)
        
        print("[DEBUG] Data loaded successfully:")
        print(f" - X shape: {X_train.shape}")
        print(f" - y shape: {y_train.shape}")

        # --- Convert data to PyTorch tensors ---
        print("[DEBUG] Converting data to tensors...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        # --- Initialize model ---
        print("[DEBUG] ===== Loading model parameters =====")
        model = load_parameters(in_model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # --- Load existing parameters if available ---
        if os.path.exists(in_model_path):
            print("[DEBUG] Loading existing model parameters...")
            load_parameters(model, in_model_path)
            
        # --- seting up the loading metadata ---
        metadata = load_metadata(in_model_path)

        # --- Training setup ---
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.MSELoss()
        
        n_batches = len(X_train_tensor) // BATCH_SIZE
        print(f"[DEBUG] Starting training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, total_batches={n_batches}")

        # this is where you can decide on how the continous mean data is loaded
        # like a parameter which specifies ensures that after each training round "continue and use the next 14 day mean data"
        
        # --- Training loop ---
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for b in range(n_batches):
                start_idx = b * BATCH_SIZE
                end_idx = start_idx + BATCH_SIZE
                
                batch_x = X_train_tensor[start_idx:end_idx].to(device)
                batch_y = y_train_tensor[start_idx:end_idx].to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if (b + 1) % 10 == 0:  # Print every 10 batches
                    print(f"[DEBUG] Epoch [{epoch+1}/{EPOCHS}] Batch [{b+1}/{n_batches}] Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / n_batches
            print(f"[DEBUG] Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

        # --- Save model and metadata ---
        print("[DEBUG] Saving model and metadata...")
        metadata = {
            "num_examples": len(X_train_tensor),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "temperature_stats": {
                "cell_max_temp": recent_stats.get("ibmu_statspktempcellmax_avg", 0),
                "cell_min_temp": recent_stats.get("ibmu_statspktempcellmin_avg", 0),
                "coolant_temp": recent_stats.get("impb_coolttemp_avg", 0)
            },
            "battery_stats": {
                "battery_voltage": recent_stats.get("ibmu_statspkvbatt_avg", 0),
                "battery_current": recent_stats.get("ibmu_statspkcurr_avg", 0),
                "state_of_health": recent_stats.get("ibmu_algopksoh_avg", 0)
            }
        }

        save_metadata(metadata, out_model_path)
        save_parameters(model, out_model_path)
        
        print(f"[DEBUG] Training completed successfully. Model saved to: {out_model_path}")

    except Exception as e:
        print("\n\n[ERROR] Training failed with exception:\n")
        traceback.print_exc()
        # Save empty metadata to avoid FEDn crashes
        save_metadata({}, out_model_path)
        

def train_linear_soh_model(X: pd.DataFrame, y: pd.Series):
    """Train a linear regression model using least squares and return weights and intercept."""
    X_aug = np.hstack([X.values, np.ones((len(X), 1))])  # add intercept term
    w_aug, *_ = np.linalg.lstsq(X_aug, y.values, rcond=None)
    weights = w_aug[:-1]
    intercept = w_aug[-1]
    return weights, intercept, X.columns.tolist()


if __name__ == "__main__":
    print("\n\n[DEBUG] ===== Starting train() via __main__ =====\n\n")
    if len(sys.argv) < 2:
        print("[ERROR] Not enough arguments provided to newTrain.py")
        sys.exit(1)
    try:
        train(sys.argv[1], sys.argv[2])
    except Exception as e:
        print("\n\n[ERROR] Something went wrong in main newTrain.py execution!\n")
        traceback.print_exc() 