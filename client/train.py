print("[DEBUG] ===== train.py is running =====")

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import json
# don't forget to update the model class name and the file to your newModel.py
from model import BatterySoHModel, load_parameters, save_parameters
#from client.model import BatterySoHModel
#from newPotentialModel import load_parameters, save_parameters

#from model import load_parameters, save_parameters
from fedn.utils.helpers.helpers import load_metadata, save_metadata
from data import load_data
import traceback
import time

data_path = os.environ.get('FEDN_DATA_PATH')
if data_path is None:
    raise ValueError("FEDN_DATA_PATH environment variable not set!")


# Configuration
BATCH_SIZE = 8  # Reduced from 32 to be smaller than dataset size
LEARNING_RATE = 0.001
EPOCHS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(in_model_path, out_model_path):
    
    ### Loading metadata
    try: 
        client_settings = load_metadata(in_model_path)
        lr = client_settings.get("learning_rate", LEARNING_RATE)
        print(f"[DEBUG] Current round: {current_round}")
        
    except Exception as e:
        print(f"[ERROR] No metadata found in {in_model_path}. Using default learning rate: {LEARNING_RATE}")
        lr = LEARNING_RATE
    
    ### Loading and preprocessing data & seting the data path
    try:
        #print(f"[DEBUG] Input model path: {in_model_path}")
        #print(f"[DEBUG] Output model path: {out_model_path}")
        #data_folder = os.environ.get('FEDN_DATA_PATH')
        current_round = 0 
        #data_file = pickDataFile(data_folder, current_round)

        # --- Load and preprocess data ---
        print("[DEBUG] Loading and preprocessing data...")
        
        # Pass window parameters to load_data function
        X_train, y_train, recent_stats = load_data(
            data_path, 
            is_train=True)
        
        print("[DEBUG] Data loaded successfully:")
        print(f"Some informaiton from recent_stats:")
        #print(recent_stats.keys())
        #print(recent_stats['feature_stats'].info())
        print(recent_stats['feature_stats'].head())
        soh_stats = recent_stats['feature_stats'].loc['SoH']
        print("SoH mean:",  soh_stats['mean'])
        print("SoH var: ",  soh_stats['var'])

        # print(f" - X shape: {X_train.shape}")
        # print(f" - X.shape[0]: {X_train.shape[0]}")
        # print(f" - X.shape[1]: {X_train.shape[1]}")
        

        # --- Convert data to PyTorch tensors ---
        #print("[DEBUG] Converting data to tensors...")
        #print(X_train.dtypes)
        
        # 1) Fill NaNs so every row has a valid number
        X_train['fast_charge']        = X_train['fast_charge'].fillna(0)
        X_train['depth_of_discharge'] = X_train['depth_of_discharge'].fillna(0)

        # 2) Convert booleans → floats (True→1.0, False→0.0)
        X_train['session_type_charge']    = X_train['session_type_charge'].astype(float)
        X_train['session_type_discharge'] = X_train['session_type_discharge'].astype(float)

        # 3) Now cast the entire DataFrame to float32
        X_train = X_train.astype('float32')

        # 4) Convert to a tensor
        X_train_tensor = torch.from_numpy(X_train.values).to(device)
        
        X_train_tensor = torch.from_numpy(X_train.values).float()
        y_train_tensor = torch.from_numpy(y_train.values).float()
        #y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        num_features = X_train.shape[1]
        
        # --- Initialize model ---
        print(f"[DEBUG] Creating model with {num_features} input features")
        model = load_parameters(in_model_path, num_features)    # Get the number of features from the data shape   
        print("[DEBUG] ===== Loading model parameters =====")
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)    
        
        # --- Training setup ---
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Use the learning rate from metadata
        criterion = torch.nn.MSELoss()
        
        #print(f"[DEBUG] Length of X_train_tensor: {len(X_train_tensor)}")
        #print(f"[DEBUG] Batch size: {BATCH_SIZE}")
        n_batches = len(X_train_tensor) // BATCH_SIZE
        if n_batches == 0: 
            n_batches = 1
        
        print(f"[DEBUG] Number of batches: {n_batches}")
        print(f"[DEBUG] Starting training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, total_batches={n_batches}")
        # --- Training loop ---
        try:
            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0
            start_time = time.time()
            
            # Progress bar setup
            progress_interval = max(1, n_batches // 10)  # Update progress every 10% of batches
            
            for b in range(n_batches):
                start_idx = b * BATCH_SIZE
                end_idx = start_idx + BATCH_SIZE
                
                batch_x = X_train_tensor[start_idx:end_idx].to(device)
                batch_y = y_train_tensor[start_idx:end_idx].to(device)

                optimizer.zero_grad()
                #print(f"[DEBUG] Batch x type: {batch_x.type()}")
                #print(f"[DEBUG] Batch x shape: {batch_x.shape}")
                # Ensure the input is properly shaped for the model
                # The model expects (batch_size, input_features)
                outputs = model(batch_x)
                #print(f"[DEBUG] Model output shape: {outputs.shape}")
                # Reshape batch_y to match outputs if needed
                if outputs.shape != batch_y.shape:
                    batch_y = batch_y.view(outputs.shape)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch_x.size(0)
                
                # Print progress every 10% of batches
                if (b + 1) % progress_interval == 0:
                    elapsed_time = time.time() - start_time
                    progress = (b + 1) / n_batches * 100
                    print(f"[DEBUG] Epoch [{epoch+1}/{EPOCHS}] Progress: {progress:.1f}% ({b+1}/{n_batches}) Loss: {loss.item():.4f} Time: {elapsed_time:.1f}s")
                    print()
            
            avg_loss = total_loss / n_batches
            epoch_time = time.time() - start_time
            print(f"[DEBUG] Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f} Time: {epoch_time:.1f}s")
        except Exception as e:
            print("--------------------------------")
            print(f"[ERROR] Error in training loop: {e}")
            print("--------------------------------")
            traceback.print_exc()
            

        # --- Save model and metadata ---
        try:
            # after calling load_data(...)
            # 1) flatten the DataFrame of means & vars into a simple dict
            stats_df   = recent_stats['feature_stats']
            flat_stats = {}
            for name, row in stats_df.iterrows():
                flat_stats[f"{name}_mean"] = float(row['mean'])
                flat_stats[f"{name}_var"]  = float(row['var'])

            # 2) add the single forecast value
            flat_stats['forecast_SoH_10'] = float(recent_stats['forecast_SoH_10'])

            # 3) assemble metadata
            metadata = {
                "num_examples":  len(X_train_tensor),
                "batch_size":    BATCH_SIZE,
                "epochs":        EPOCHS,
                "learning_rate": lr,
                "recent_stats":  flat_stats
            }

            # 4) (optional) write JSON to disk for inspection
            with open("client_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # 5) save via your helper
            print("[DEBUG] Saving model and metadata…")
            save_metadata(metadata, out_model_path)
            save_parameters(model, out_model_path)
            print(f"[DEBUG] Training completed and saving metadata and parameters successfully. Model saved to: {out_model_path}")
            
        except Exception as e:
            print(f"[ERROR] Error saving metadata or parameters: {e}")
        
        

    except Exception as e:
        print("\n\n[ERROR] Training failed with exception:\n")
        traceback.print_exc()
        # Save empty metadata to avoid FEDn crashes
        save_metadata({}, out_model_path)
        


def pickDataFile(folder, current_round):
    # Get all CSV files in the folder
    try:    
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        
        # Sort files chronologically based on year and sequence number
        def extract_date_info(filename):
            # Extract year from the filename (e.g., "2024" or "2025")
            year = filename.split('-')[-1].replace('.csv', '')
            # Extract sequence number (e.g., "1", "2", "10", etc.)
            seq_num = filename.split('_')[-1].split('-')[0]
            # Convert to integers for proper sorting
            return int(year), int(seq_num)
        
        # Sort the files based on the extracted date information
        csv_files.sort(key=extract_date_info)
        
        selected_file = csv_files[current_round]
        return os.path.join(folder, selected_file)
        
    except Exception as e:
        print(f"[ERROR] Error getting CSV files in {folder}: {e}")
        return None
    


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
        print("[ERROR] Not enough arguments provided to train.py")
        sys.exit(1)
    try:
        train(sys.argv[1], sys.argv[2])
    except Exception as e:
        print("\n\n[ERROR] Something went wrong in main train.py execution!\n")
        traceback.print_exc() 