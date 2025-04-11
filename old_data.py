# old data.py


import os
from math import floor
import sys

import torch
import torchvision
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys


dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


FEATURE_COLUMNS = [
    'ibmu_statspkblkvoltavg', #1
    'ibmu_statspkcurr', #2
    'ibmu_algopksoctrue', #3
    'ibmu_statspktempcellmax', #4
    'ibmu_statspktempcellmin', #5
    'impb_coolttemp', #6
    'ivcu_ambairtemp', #7
    'ibmu_wntytotkwhrgn' #8

]
TARGET_COLUMN = 'ibmu_algopksoh'


def data_init():
    """
    Basic data initializer for FEDn 'startup' step.
    - Reads the DATA_PATH environment variable
    - Logs the size of the CSV
    - Does not modify or split the data
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

#if __name__ == "__main__":
#    data_init()



# Load DATA_PATH from environment variable
data_path = os.environ.get('FEDN_DATA_PATH')
if data_path is None:
    raise ValueError("FEDN_DATA_PATH environment variable not set!")
print(f"[DEBUG] data.py running with sys.argv: {sys.argv}")

def estimate_soh(cycles, temp, fast_charges):
    """ Compute State of Health (SoH) based on charge cycles, temperature, and fast charges. """
    return np.maximum(100 - 0.03 * cycles - 0.01 * (cycles**1.2) - 2 * np.exp(-0.005 * temp) - 0.5 * fast_charges, 50)




def load_data(filepath, is_train=True):
    """Load and preprocess the Indian Fleet dataset for a specific EV client."""
    print(f"[DEBUG] Loading data from {filepath}")

    # check if the data is from the Indian Fleet Data
    if "IndianFleetData" in filepath:
        try:
            chunk_size = 80000
            chunk_iter = pd.read_csv(filepath, chunksize=chunk_size)
            df = next(chunk_iter)

            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True, errors='coerce')
            df = df.sort_values('timestamp')
            df = df.dropna(subset=["timestamp"])

            time_cutoff = df['timestamp'].max() - pd.Timedelta(days=14)
            recent_data = df[df['timestamp'] > time_cutoff]
            recent_stats = {
                "temperature_avg": recent_data["batTemp"].mean() if not recent_data.empty else 0.0,
                "batMaxTemp_avg": recent_data["batMaxTemp"].mean() if not recent_data.empty else 0.0,
                "socPercent_avg": recent_data["socPercent"].mean() if not recent_data.empty else 0.0,
                "chargeCycle_std": recent_data["chargeCycle"].std() if not recent_data.empty else 0.0
            }

            feature_columns = ["chargeCycle", "batVolt", "batCurrent", "socPercent", "batTemp", "batMaxTemp"]
            df = df.dropna(subset=feature_columns)

            if "fastChargeCount" in df.columns:
                fast_charges = df["fastChargeCount"].values
            else:
                fast_charges = np.random.randint(0, 5, size=len(df))

            X = df[feature_columns].values.astype(np.float32)
            y = estimate_soh(df["chargeCycle"].values, df["batTemp"].values, fast_charges)

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

    # if the data is not from the Indian Fleet Data, then it is from the US Fleet Data
    else:
        try:
            chunk_size = 80000
            chunk_iter = pd.read_csv(filepath, chunksize=chunk_size)
            df = next(chunk_iter)

            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True, errors='coerce')            
            df = df.sort_values('timestamp')
            df_ma = df[FEATURE_COLUMNS].rolling(window='14D', min_periods=1).mean()
            df[FEATURE_COLUMNS] = df_ma
            df = df.dropna(subset=["timestamp"])

            time_cutoff = df['timestamp'].max() - pd.Timedelta(days=14)
            recent_data = df[df['timestamp'] > time_cutoff]
            
            '''recent_stats = {
                "temperature_avg": recent_data["batTemp"].mean() if not recent_data.empty else 0.0,
                "batMaxTemp_avg": recent_data["batMaxTemp"].mean() if not recent_data.empty else 0.0,
                "socPercent_avg": recent_data["socPercent"].mean() if not recent_data.empty else 0.0,
                "chargeCycle_std": recent_data["chargeCycle"].std() if not recent_data.empty else 0.0
            }'''
            
            recent_stats = {
                "ambairtemp": recent_data["ivcu_ambairtemp"].mean() if not recent_data.empty else 0.0,
                "ibmu_statspkblkvoltavg_avg": recent_data["ibmu_statspkblkvoltavg"].mean() if not recent_data.empty else 0.0,
                "ibmu_statspkcurr_avg": recent_data["ibmu_statspkcurr"].mean() if not recent_data.empty else 0.0,
                "ibmu_algopksoctrue_avg": recent_data["ibmu_algopksoctrue"].mean() if not recent_data.empty else 0.0,
                "ibmu_statspktempcellmax": recent_data["ibmu_statspktempcellmax"] if not recent_data.empty else 0.0,
                "ibmu_statspktempcellmin": recent_data["ibmu_statspktempcellmin"] if not recent_data.empty else 0.0,
                "impb_coolttemp_avg": recent_data["impb_coolttemp"].mean() if not recent_data.empty else 0.0,
                "ibmu_wntytotkwhrgn_avg": recent_data["ibmu_wntytotkwhrgn"].mean() if not recent_data.empty else 0.0
            }
            
            feature_columns = FEATURE_COLUMNS
            df = df.dropna(subset=feature_columns)

            # standardize the data
            X = df[feature_columns].values.astype(np.float32)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y = df[TARGET_COLUMN].values.astype(np.float32)
            
            # check if the data is being used for training or evaluation    
            if is_train:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)
                print(f"[DEBUG] Loaded {len(X_train)} training samples, {len(X_test)} test samples.")
                return X_train, y_train, recent_stats
            else:
                print(f"[DEBUG] Loaded {len(X_scaled)} samples for evaluation.")
                return X_scaled, y, recent_stats
        
        # if there is an error, print the error and raise it
        except Exception as e:
            print(f"[ERROR] Failed to load or process data from {filepath}")
            print(f"[DEBUG] Exception: {e}")
            raise e


# if the file is run directly, print the data path and load the data
if __name__ == "__main__":
    #print("\n\n[DEBUG] ===== Starting load_data() via __main__ =====\n\n")
    data_path = os.getenv("FEDN_DATA_PATH")
    if data_path is None:
        print("[ERROR] FEDN_DATA_PATH environment variable is not set!")
        exit(1)
    if not os.path.exists(data_path):
        print(f"[ERROR] File not found at FEDN_DATA_PATH: {data_path}")
        exit(1)

    print(f"[DEBUG] Using dataset at: {data_path}")

    try:
        X, y, recent_stats = load_data(data_path)
        print(f"[DEBUG] ✅ Data loaded successfully!")
        print(f" - X shape: {X.shape}")
        #print(f" - y shape: {y.shape}")
    except Exception as e:
        print("[ERROR] Something went wrong in main data.py execution!")
    
    
    
        

