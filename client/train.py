print("[DEBUG] ===== train.py is running =====")

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
# don't forget to update the model class name and the file to your newModel.py
from model import BatterySoHModel, load_parameters, save_parameters
#from client.model import BatterySoHModel
#from newPotentialModel import load_parameters, save_parameters

#from model import load_parameters, save_parameters
from fedn.utils.helpers.helpers import load_metadata, save_metadata
from data import load_data
import traceback
import time

# Configuration
BATCH_SIZE = 8  # Reduced from 32 to be smaller than dataset size
LEARNING_RATE = 0.001
EPOCHS = 2

def train(in_model_path, out_model_path):
    
    ### Loading metadata
    try: 
        client_settings = load_metadata(in_model_path)
        lr = client_settings.get("learning_rate", LEARNING_RATE)
        orch_params = client_settings.get("data_orch_params", {})
        current_round = orch_params.get("current_round", 0)
        print(f"[DEBUG] Current round: {current_round}")
        # Extract window parameters for data orchestration
        window_offset = orch_params.get("window_offset", 0)
        number_of_cycles_to_compare = orch_params.get("number_of_cycles_to_compare", 10) # this could be tested out with and compared with different numbers to find the optimal one)
        #window_length = orch_params.get("window_length", 14)
        print(f"[DEBUG] Using data orchestration parameters: offset={window_offset}, number_of_cycles_to_compare={number_of_cycles_to_compare}")

    except Exception as e:
        print(f"[ERROR] No metadata found in {in_model_path}. Using default learning rate: {LEARNING_RATE}")
        lr = LEARNING_RATE
        window_offset = 0
        number_of_cycles_to_compare = 10
        #window_length = 14
    
    
    ### Loading and preprocessing data & seting the data path
    try:
        #print(f"[DEBUG] Input model path: {in_model_path}")
        #print(f"[DEBUG] Output model path: {out_model_path}")
        data_folder = os.environ.get('FEDN_DATA_PATH')
        current_round = 0 
        data_file = pickDataFile(data_folder, current_round)
        # --- Load and preprocess data ---
        print("[DEBUG] Loading and preprocessing data...")
        
        # Pass window parameters to load_data function
        X_train, y_train, recent_stats = load_data(
            data_file, 
            is_train=True, 
            window_offset=window_offset, 
            number_of_cycles_to_compare = number_of_cycles_to_compare
            #window_length=window_length
        )
        
        print("[DEBUG] Data loaded successfully:")
        print(f"Some informaiton on the data:")
        print(f" - X shape: {X_train.shape}")
        print(f" - X.shape[0]: {X_train.shape[0]}")
        print(f" - X.shape[1]: {X_train.shape[1]}")
        #print(f" - y shape: {y_train.shape}")
        #print(f" - Recent stats: {recent_stats}")

        # --- Convert data to PyTorch tensors ---
        #print("[DEBUG] Converting data to tensors...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        num_features = X_train.shape[1]
        
        # --- Initialize model ---
        print(f"[DEBUG] Creating model with {num_features} input features")
        model = load_parameters(in_model_path, num_features)    # Get the number of features from the data shape   
        print("[DEBUG] ===== Loading model parameters =====")
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
            print("[DEBUG] Saving model and metadata...")
            metadata = {
                "num_examples": len(X_train_tensor),
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": lr,
                "temperature_stats": {
                    "cell_max_temp": recent_stats.get("ibmu_statspktempcellmax", 0),
                    "cell_min_temp": recent_stats.get("ibmu_statspktempcellmin", 0),
                    "coolant_temp": recent_stats.get("impb_coolttemp_avg", 0)
                },
                "battery_stats": {
                    "battery_voltage": recent_stats.get("ibmu_statspkvbatt", 0),
                    "battery_current": recent_stats.get("ibmu_statspkcurr_avg", 0),
                    "state_of_health": recent_stats.get("ibmu_algopksoh_avg", 0)
                },
                # Add driving behavior stats to metadata
                "driving_behavior": {
                    "rms_current_1h": recent_stats.get("rms_current_1h_avg", 0),
                    "rms_current_1d": recent_stats.get("rms_current_1d_avg", 0),
                    "max_acceleration": recent_stats.get("max_acceleration_avg", 0),
                    "avg_speed": recent_stats.get("avg_speed_avg", 0),
                    "driving_aggressiveness": recent_stats.get("driving_aggressiveness_avg", 0),
                    "battery_stress": recent_stats.get("battery_stress_avg", 0)
                },
                # Include data orchestration parameters in metadata
                "data_orch_params": {
                    "window_offset": window_offset,
                    "window_length": window_length
                }
            }
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