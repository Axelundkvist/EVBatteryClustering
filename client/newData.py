import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

# Set up paths
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

def data_init():
    """
    Basic data initializer for FEDn 'startup' step.
    - Reads the FEDN_DATA_PATH environment variable.
    - Logs the size of the CSV.
    """
    data_path = os.environ.get("FEDN_DATA_PATH")
    if not data_path:
        print("[ERROR] data.py: FEDN_DATA_PATH is not set!")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"[ERROR] data.py: File not found at {data_path}")
        sys.exit(1)
    try:
        chunk_iter = pd.read_csv(data_path, chunksize=80000)
        df = next(chunk_iter)
        print(f"[INFO] data.py: Successfully loaded {len(df)} rows from {data_path}.")
    except Exception as e:
        print(f"[ERROR] data.py: Failed to read CSV from {data_path}")
        print(f"[DEBUG] Exception: {e}")
        sys.exit(1)

# Define new feature set based on dataset analysis
FEATURE_COLUMNS = [
    "iaccel_long", "ibmu_algopksoctrue", "ibmu_algopksoh", "ibmu_statspkvbatt",
    "ibmu_statspkcurr", "ibmu_statspkblkvoltavg", "ibmu_statspkblkvoltdelta",
    "ibmu_statspktempcellmax", "ibmu_statspktempcellmin", "impb_coolttemp",
    "ivcu_battcoolflowrate", "ivehspd", "ibmu_wntytotkwhrgn"
]

def estimate_soh(cycles, temp, fast_charges):
    """Compute State of Health (SoH) based on charge cycles, temperature, and fast charges."""
    return np.maximum(100 - 0.03 * cycles - 0.01 * (cycles**1.2) - 2 * np.exp(-0.005 * temp) - 0.5 * fast_charges, 50)

def load_data(filepath, is_train=True):
    """
    Load and preprocess the EV battery dataset for a specific EV client.
    
    This function:
      - Loads a chunk of the CSV file.
      - Parses the 'date' (and optionally 'time') columns to create a 'timestamp'.
      - Sorts the data by timestamp.
      - Computes a 14-day sliding window average for the selected features.
      - Drops rows with missing values in the key feature columns.
      - Splits data into training and testing sets.
    """
    print(f"[DEBUG] Loading data from {filepath}")
    
    try:
        chunk_size = 80000
        chunk_iter = pd.read_csv(filepath, chunksize=chunk_size)
        df = next(chunk_iter)
        
        # Combine date and time if time column exists, otherwise use date only.
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True, errors='coerce')
        else:
            df['timestamp'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.sort_values('timestamp')
        df = df.dropna(subset=["timestamp"])
        
        # Set timestamp as index for rolling window calculations
        df.set_index('timestamp', inplace=True)
        
        # Compute the 14-day sliding moving average for selected features.
        # We assume the data frequency allows a 14-day window; adjust the window if needed.
        df_ma = df[FEATURE_COLUMNS].rolling(window='14D', min_periods=1).mean()
        
        # Optional: you could merge the moving average back to the original dataframe
        # or use it as your primary feature set.
        df[FEATURE_COLUMNS] = df_ma
        
        # Drop any rows where key features are still missing
        df = df.dropna(subset=FEATURE_COLUMNS)
        
        # Reset index to bring timestamp back as a column if needed downstream
        df.reset_index(inplace=True)
        
        # For demonstration, compute recent stats over the last 14 days
        time_cutoff = df['timestamp'].max() - pd.Timedelta(days=14)
        recent_data = df[df['timestamp'] > time_cutoff]
        recent_stats = {
            "iaccel_long_avg": recent_data["iaccel_long"].mean(),
            "ibmu_algopksoh_avg": recent_data["ibmu_algopksoh"].mean(),
            "ibmu_statspkvbatt_avg": recent_data["ibmu_statspkvbatt"].mean()
            # Add more if needed
        }
        
        # Ensure fast charge count exists, if not, simulate it
        if "fastChargeCount" in df.columns:
            fast_charges = df["fastChargeCount"].values
        else:
            fast_charges = np.random.randint(0, 5, size=len(df))
        
        # Prepare feature matrix X and target y (using charge cycles for SoH estimation as an example)
        # If your new target is different, update accordingly.
        if "chargeCycle" in df.columns:
            cycles = df["chargeCycle"].values
        else:
            cycles = np.zeros(len(df))
        
        # Use one of the temperature features, here 'ibmu_statspktempcellmax'
        if "ibmu_statspktempcellmax" in df.columns:
            temp = df["ibmu_statspktempcellmax"].values
        else:
            temp = np.zeros(len(df))
        
        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = estimate_soh(cycles, temp, fast_charges)
        
        if is_train:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            print(f"[DEBUG] Loaded {len(X_train)} training samples, {len(X_test)} test samples.")
            return X_train, y_train, recent_stats
        else:
            print(f"[DEBUG] Loaded {len(X)} samples for evaluation.")
            return X, y, recent_stats
        
    except Exception as e:
        print(f"[ERROR] Failed to load or process data from {filepath}")
        print(f"[DEBUG] Exception: {e}")
        raise e

def check_training_eligibility(filepath, temp_threshold):
    """
    Check if a client should participate in training based on temperature conditions.
    
    Uses the 14-day mean of 'ibmu_statspktempcellmax' as a proxy for temperature.
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    last_14_days = df.sort_values(by='date', ascending=False).head(14)
    mean_temp = last_14_days['ibmu_statspktempcellmax'].mean()
    return mean_temp > temp_threshold

if __name__ == "__main__":
    data_path = os.getenv("FEDN_DATA_PATH")
    if data_path is None:
        print("[ERROR] FEDN_DATA_PATH environment variable is not set!")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"[ERROR] File not found at FEDN_DATA_PATH: {data_path}")
        sys.exit(1)
    
    print(f"[DEBUG] Using dataset at: {data_path}")
    try:
        X, y, stats = load_data(data_path)
        print(f"[DEBUG] ✅ Data loaded successfully!")
        print(f" - X shape: {X.shape}")
        print(f" - Recent stats: {stats}")
    except Exception as e:
        print("[ERROR] Something went wrong in main data.py execution!")
