import os
from math import floor
import sys

import torch
import torchvision
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


import os
import sys
import pandas as pd

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
print(f"WOAH - MASSIVE MESSAGE FROM DATA.PY PLZ WORK")


def estimate_soh(cycles, temp, fast_charges):
    """ Compute State of Health (SoH) based on charge cycles, temperature, and fast charges. """
    return np.maximum(100 - 0.03 * cycles - 0.01 * (cycles**1.2) - 2 * np.exp(-0.005 * temp) - 0.5 * fast_charges, 50)


def load_data(filepath, is_train=True):
    """Load and preprocess the Indian Fleet dataset for a specific EV client."""
    print(f"[DEBUG] Loading data from {filepath}")

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


def check_training_eligibility(filepath, temp_threshold):
    """ Check if a client should participate in training based on temperature conditions. """
    df = pd.read_csv(filepath)
    
    # Compute 14-day mean temperature
    df['date'] = pd.to_datetime(df['date'])
    last_14_days = df.sort_values(by='date', ascending=False).head(14)
    mean_temp = last_14_days['batMaxTemp'].mean()

    return mean_temp > temp_threshold  # Returns True if threshold exceeded



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
        X, y = load_data(data_path)
        print(f"[DEBUG] ✅ Data loaded successfully!")
        print(f" - X shape: {X.shape}")
        #print(f" - y shape: {y.shape}")
    except Exception as e:
        print("[ERROR] Something went wrong in main data.py execution!")
    
    
    
        
'''if __name__ == "__main__":
print("\n\n[DEBUG] ===== Starting train() via __main__ =====\n\n")
if len(sys.argv) < 3:
    print("[ERROR] Not enough arguments provided to train.py")
    sys.exit(1)
try:
    train(sys.argv[1], sys.argv[2], data_path)
except Exception as e:
    print("\n\n[ERROR] Something went wrong in main train.py execution!\n")
    traceback.print_exc()'''


'''
def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/1/mnist.pt")

    data = torch.load(data_path, weights_only=True)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    # Normalize
    X = X / 255

    return X, y
'''

'''def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result
'''

'''def split(out_dir="data"):
    n_splits = int(os.environ.get("FEDN_NUM_DATA_SPLITS", 2))

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    # Load and convert to dict
    train_data = torchvision.datasets.MNIST(root=f"{out_dir}/train", transform=torchvision.transforms.ToTensor, train=True)
    test_data = torchvision.datasets.MNIST(root=f"{out_dir}/test", transform=torchvision.transforms.ToTensor, train=False)
    data = {
        "x_train": splitset(train_data.data, n_splits),
        "y_train": splitset(train_data.targets, n_splits),
        "x_test": splitset(test_data.data, n_splits),
        "y_test": splitset(test_data.targets, n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save(
            {
                "x_train": data["x_train"][i],
                "y_train": data["y_train"][i],
                "x_test": data["x_test"][i],
                "y_test": data["y_test"][i],
            },
            f"{subdir}/mnist.pt",
        )
'''



'''

to train things locally try and do things like

python train.py ../seed.npz ../model_update.npz --data_path data/clients/1/mnist.pt
python validate.py ../model_update.npz ../validation.json --data_path data/clients/1/mnist.pt


python train.py ../seed.npz ../model_update.npz --data_path "/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_54309277.csv"




'''