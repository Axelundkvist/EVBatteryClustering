import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Set up paths
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)



# Define new feature set based on dataset analysis
FEATURE_COLUMNS = [
    "iaccel_long", "ibmu_algopksoctrue", "ibmu_statspkvbatt",
    "ibmu_statspkcurr", "ibmu_statspkblkvoltavg", "ibmu_statspkblkvoltdelta",
    "ibmu_statspktempcellmax", "ibmu_statspktempcellmin", "impb_coolttemp",
    "ivcu_battcoolflowrate", "ivehspd", "ibmu_wntytotkwhrgn"
]

# Map lowercase feature names to their uppercase equivalents
FEATURE_COLUMN_MAPPING = {
    "iaccel_long": "IAccel_Long",
    "ibmu_algopksoctrue": "IBMU_AlgoPkSocTrue",
    "ibmu_statspkvbatt": "IBMU_StatsPkVbatt",
    "ibmu_statspkcurr": "IBMU_StatsPkCurr",
    "ibmu_statspkblkvoltavg": "IBMU_StatsPkBlkVoltAvg",
    "ibmu_statspkblkvoltdelta": "IBMU_StatsPkBlkVoltDelta",
    "ibmu_statspktempcellmax": "IBMU_StatsPkTempCellMax",
    "ibmu_statspktempcellmin": "IBMU_StatsPkTempCellMin",
    "impb_coolttemp": "IMPB_CooltTemp",
    "ivcu_battcoolflowrate": "IVCU_BattCoolFlowRate",
    "ivehspd": "IVehSpd",
    "ibmu_wntytotkwhrgn": "IBMU_WntyTotKWhRgn"
}

# New driving behavior features to be calculated
DRIVING_BEHAVIOR_FEATURES = [
    "rms_current_1h",      # RMS current over 1-hour window
    "rms_current_1d",      # RMS current over 1-day window
    "max_acceleration",    # Maximum acceleration
    "avg_speed",           # Average speed
    "speed_variance",      # Speed variance
    "charge_frequency",    # Frequency of charging events
    "discharge_rate",      # Rate of discharge
    "temperature_range",   # Temperature range (max-min)
    "battery_stress",      # Battery stress indicator
    "driving_aggressiveness" # Aggressiveness score
]

# TODO: change the fast_charges to a parameter derived from the current parameters
# TODO: use the actual SoH parameter as the target
# TODO: find an article that acutally derives a SoH model from the data
def estimate_soh(cycles, temp, fast_charges): # change this to the new target
    """Compute State of Health (SoH) based on charge cycles, temperature, and fast charges."""
    return np.maximum(100 - 0.03 * cycles - 0.01 * (cycles**1.2) - 2 * np.exp(-0.005 * temp) - 0.5 * fast_charges, 50)

def calculate_rms_current(df, window='1h'):
    """
    Calculate RMS current over a specified time window.
    
    Args:
        df: DataFrame with IBMU_StatsPkCurr column and datetime index
        window: Time window for calculation (default: '1h')
        
    Returns:
        Series with RMS current values
    """
    print(f"[DEBUG] calculate_rms_current: DataFrame index type: {type(df.index)}")
    
    if 'IBMU_StatsPkCurr' not in df.columns:
        print("[WARNING] Column 'IBMU_StatsPkCurr' not found. Returning zeros.")
        return pd.Series(0, index=df.index)
    
    try:
        # Convert window to pd.Timedelta if it's a string
        if isinstance(window, str):
            window = pd.Timedelta(window)
            print(f"[DEBUG] calculate_rms_current: Converted window '{window}' to Timedelta")
        
        # Optimize: Use a more efficient approach for large datasets
        # For very large datasets, we can use a resampling approach instead of rolling
        if len(df) > 10000:
            print(f"[DEBUG] calculate_rms_current: Using resampling for large dataset ({len(df)} rows)")
            # Resample to reduce the number of calculations
            resampled = df['IBMU_StatsPkCurr'].resample(window).apply(
                lambda x: np.sqrt(np.mean(x**2)) if len(x) > 0 else 0
            )
            # Reindex to match the original index
            return resampled.reindex(df.index, method='ffill')
        else:
            # For smaller datasets, use the rolling approach
            return df['IBMU_StatsPkCurr'].rolling(window=window).apply(
                lambda x: np.sqrt(np.mean(x**2)) if len(x) > 0 else 0
            )
    except Exception as e:
        print(f"[WARNING] Error calculating RMS current: {e}. Using zeros instead.")
        print(f"[DEBUG] calculate_rms_current: Exception details: {str(e)}")
        return pd.Series(0, index=df.index)

def calculate_driving_behavior_features(df):
    """
    Calculate driving behavior features from the dataset.
    
    Args:
        df: DataFrame with datetime index and relevant columns
        
    Returns:
        DataFrame with driving behavior features
    """
    print(f"[DEBUG] calculate_driving_behavior_features: DataFrame index type: {type(df.index)}")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_features = df.copy()
    
    # Check if we have the required column
    if 'IBMU_StatsPkCurr' not in df.columns:
        print("[WARNING] Column 'IBMU_StatsPkCurr' not found. Returning zeros for all features.")
        # Initialize all features with zeros
        for feature in DRIVING_BEHAVIOR_FEATURES:
            df_features[feature] = 0
        return df_features
    
    # Optimize: Calculate all features in a single pass if possible
    try:
        print("[DEBUG] calculate_driving_behavior_features: Calculating features efficiently")
    
    # Calculate RMS current over different time windows
        print("[DEBUG] calculate_driving_behavior_features: Calculating RMS current")
        df_features['rms_current_1h'] = calculate_rms_current(df, window='1h')
        df_features['rms_current_1d'] = calculate_rms_current(df, window='1d')
        
        # For large datasets, use resampling instead of rolling for better performance
        if len(df) > 10000:
            print("[DEBUG] calculate_driving_behavior_features: Using resampling for large dataset")
            
            # Resample to 1-hour intervals for 1-hour features
            hourly_resampled = df['IBMU_StatsPkCurr'].resample('1h')
            
            # Calculate features from resampled data
            df_features['max_acceleration'] = hourly_resampled.max().reindex(df.index, method='ffill')
            df_features['avg_speed'] = hourly_resampled.mean().reindex(df.index, method='ffill')
            df_features['speed_variance'] = hourly_resampled.var().reindex(df.index, method='ffill')
            df_features['temp_range'] = hourly_resampled.max().reindex(df.index, method='ffill') - \
                                       hourly_resampled.min().reindex(df.index, method='ffill')
            
            # Resample to 1-day intervals for daily features
            daily_resampled = df['IBMU_StatsPkCurr'].resample('1d')
            
            # Calculate daily features
            df_features['charge_frequency'] = daily_resampled.apply(
                lambda x: np.sum(x > 0) if len(x) > 0 else 0
            ).reindex(df.index, method='ffill')
            
            df_features['discharge_rate'] = daily_resampled.apply(
                lambda x: np.mean(x[x < 0]) if len(x[x < 0]) > 0 else 0
            ).reindex(df.index, method='ffill')
        else:
            # For smaller datasets, use the rolling approach
            print("[DEBUG] calculate_driving_behavior_features: Using rolling for smaller dataset")
            
            # Calculate maximum acceleration over 1-hour window
            print("[DEBUG] calculate_driving_behavior_features: Calculating max acceleration")
            df_features['max_acceleration'] = df['IBMU_StatsPkCurr'].rolling(window=pd.Timedelta('1h')).max()
            
            # Calculate average speed and variance over 1-hour window
            print("[DEBUG] calculate_driving_behavior_features: Calculating speed metrics")
            df_features['avg_speed'] = df['IBMU_StatsPkCurr'].rolling(window=pd.Timedelta('1h')).mean()
            df_features['speed_variance'] = df['IBMU_StatsPkCurr'].rolling(window=pd.Timedelta('1h')).var()
            
            # Calculate charge frequency (number of charging events per day)
            print("[DEBUG] calculate_driving_behavior_features: Calculating charge frequency")
            df_features['charge_frequency'] = df['IBMU_StatsPkCurr'].rolling(window=pd.Timedelta('1d')).apply(
                lambda x: np.sum(x > 0) if len(x) > 0 else 0
            )
            
            # Calculate discharge rate (average current when discharging)
            print("[DEBUG] calculate_driving_behavior_features: Calculating discharge rate")
            df_features['discharge_rate'] = df['IBMU_StatsPkCurr'].rolling(window=pd.Timedelta('1d')).apply(
                lambda x: np.mean(x[x < 0]) if len(x[x < 0]) > 0 else 0
            )
            
            # Calculate temperature range over 1-hour window
            print("[DEBUG] calculate_driving_behavior_features: Calculating temperature range")
            df_features['temp_range'] = df['IBMU_StatsPkCurr'].rolling(window=pd.Timedelta('1h')).max() - \
                                       df['IBMU_StatsPkCurr'].rolling(window=pd.Timedelta('1h')).min()
        
        # Calculate battery stress (combination of high current and temperature)
        print("[DEBUG] calculate_driving_behavior_features: Calculating battery stress")
        df_features['battery_stress'] = df_features['rms_current_1h'] * df_features['temp_range']
    
    # Calculate driving aggressiveness score
        print("[DEBUG] calculate_driving_behavior_features: Calculating driving aggressiveness")
        # Normalize acceleration and speed
        norm_acceleration = (df_features['max_acceleration'] - df_features['max_acceleration'].min()) / \
                          (df_features['max_acceleration'].max() - df_features['max_acceleration'].min() + 1e-6)
        norm_speed = (df_features['avg_speed'] - df_features['avg_speed'].min()) / \
                    (df_features['avg_speed'].max() - df_features['avg_speed'].min() + 1e-6)
        
        # Calculate aggressiveness score (weighted combination)
        df_features['driving_aggressiveness'] = 0.6 * norm_acceleration + 0.4 * norm_speed
        
    except Exception as e:
        print(f"[WARNING] Error calculating driving behavior features: {e}")
        # Initialize all features with zeros in case of error
        for feature in DRIVING_BEHAVIOR_FEATURES:
            df_features[feature] = 0
    
    # Fill NaN values with 0
    df_features = df_features.fillna(0)
    
    return df_features

def load_data(filepath, is_train=True, window_offset=0, window_length=14):
    """
    Load and preprocess the EV battery dataset for a specific EV client.
    
    This function:
      - Loads a chunk of the CSV file.
      - Parses the 'timestamp' column to create a datetime index.
      - Sorts the data by timestamp.
      - Computes a sliding window average for the selected features.
      - Calculates additional driving behavior features.
      - Drops rows with missing values in the key feature columns.
      - Splits data into training and testing sets.
      
    Args:
        filepath: Path to the data file
        is_train: Whether to return training or testing data
        window_offset: Offset for the data window (for orchestration)
        window_length: Length of the data window in days
    """
    print(f"[DEBUG] Loading data from {filepath}")
    # seting reminder to to rename the file to data.py later
    file_name = os.path.basename('/Users/Axel/fedn/examples/mnist-pytorch/newData.py')
    print(f"[DEBUG] using {file_name} to load data")
    print(f"[DEBUG] Using window offset: {window_offset}, window length: {window_length}")

    try:
        # Get chunk size from an environment variable, with 80000 as the default.
        # option 1:
        default_chunk_size = 100000
        chunk_size = int(os.environ.get("CHUNK_SIZE", default_chunk_size))
        # option 2:
        file_size_bytes = os.path.getsize(filepath)
        # If the file size is below a threshold (e.g., 100 MB), load the entire file.
        if file_size_bytes < 100 * 1024 * 1024:  # 100 MB
            print("[INFO] File is small, loading entire file into memory.")
            df = pd.read_csv(filepath)
        else:
            print(f"[INFO] Large file detected ({file_size_bytes} bytes), using chunk size {chunk_size}.")
            # For large files, we can optimize by only loading the columns we need
            needed_columns = ['timestamp'] + [col for col in FEATURE_COLUMNS if col in FEATURE_COLUMN_MAPPING.values()]
            chunk_iter = pd.read_csv(filepath, chunksize=chunk_size, usecols=needed_columns)
            df = next(chunk_iter)
        
        
        # comment this stuff out later
        # Debug prints to understand the DataFrame structure
        print(f"[DEBUG] DataFrame index type: {type(df.index)}")
        print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
        if 'timestamp' in df.columns:
            print(f"[DEBUG] 'timestamp' column type: {df['timestamp'].dtype}")
            print(f"[DEBUG] First few values of 'timestamp': {df['timestamp'].head()}")
        
        # Check if the data already has a timestamp index
        if isinstance(df.index, pd.DatetimeIndex):
            print("[INFO] Data already has a DatetimeIndex, using it directly.")
            # No need to create timestamp column
        else:
            # Check if timestamp column exists
            if 'timestamp' in df.columns:
                print("[INFO] Converting 'timestamp' column to datetime index.")
                # Convert timestamp column to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                print(f"[DEBUG] After conversion, 'timestamp' column type: {df['timestamp'].dtype}")
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Drop rows with NaN timestamps
                df = df.dropna(subset=["timestamp"])
                
                # Set timestamp as index for rolling window calculations
                df.set_index('timestamp', inplace=True)
                print(f"[DEBUG] After setting index, DataFrame index type: {type(df.index)}")
            # Create timestamp from date/time columns if they exist
            elif 'date' in df.columns:
                if 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True, errors='coerce')
                else:
                    df['timestamp'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                df = df.sort_values('timestamp')
                df = df.dropna(subset=["timestamp"])
                # Set timestamp as index for rolling window calculations
                df.set_index('timestamp', inplace=True)
            else:
                print("[WARNING] No timestamp or date column found in the data. Creating a dummy timestamp index.")
                # Create a dummy timestamp index if none exists
                df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1min')
                df.set_index('timestamp', inplace=True)
        
        # Optimize: Apply window offset and length before calculating features
        # This reduces the amount of data we need to process
        if window_offset > 0 or window_length > 0:
            # Get the date range
            date_range = df.index.unique()
            if len(date_range) > window_offset + window_length:
                # Select the appropriate window of dates
                selected_dates = date_range[window_offset:window_offset+window_length]
                df = df[df.index.isin(selected_dates)]
                print(f"[DEBUG] Selected data window: {selected_dates[0]} to {selected_dates[-1]}")
        
        # Calculate additional driving behavior features
        print("[DEBUG] Calculating driving behavior features...")
        df = calculate_driving_behavior_features(df)
        
        # Compute the sliding window average for selected features
        # We assume the data frequency allows a window; adjust if needed.
        
        # Check which feature columns exist in the dataset
        available_features = []
        for feature in FEATURE_COLUMNS:
            # Try to find the column in the dataset (case-insensitive)
            if feature in df.columns:
                available_features.append(feature)
            elif feature.upper() in df.columns:
                available_features.append(feature.upper())
            elif FEATURE_COLUMN_MAPPING.get(feature) in df.columns:
                available_features.append(FEATURE_COLUMN_MAPPING.get(feature))
        
        if not available_features:
            print("[WARNING] None of the specified feature columns found in the dataset.")
            print(f"[DEBUG] Available columns: {df.columns.tolist()}")
            # Use all numeric columns as features if no specified features are found
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                print(f"[INFO] Using all numeric columns as features: {numeric_cols}")
                available_features = numeric_cols
            else:
                print("[ERROR] No numeric columns found in the dataset.")
                raise ValueError("No numeric columns found in the dataset.")
        
        try:
            # Convert window_length to a proper time-based window
            if isinstance(window_length, int) and window_length > 0:
                # For time-based data, convert days to a proper time window
                window_timedelta = pd.Timedelta(days=window_length)
                # Use the time-based window for rolling
                df_ma = df[available_features].rolling(window=window_timedelta, min_periods=1).mean()
            else:
                # If window_length is not a positive integer, use a default value
                print(f"[WARNING] Invalid window_length: {window_length}. Using default value of 1 day.")
                window_timedelta = pd.Timedelta(days=1)
                df_ma = df[available_features].rolling(window=window_timedelta, min_periods=1).mean()
        except Exception as e:
            print(f"[WARNING] Error in rolling window calculation: {e}. Using simple mean instead.")
            # Fallback to simple mean if rolling fails
            df_ma = df[available_features].mean().to_frame().T
            # Repeat the mean for each row in the original dataframe
            df_ma = pd.concat([df_ma] * len(df), ignore_index=True)
            df_ma.index = df.index
        
        # Add the driving behavior features
        for feature in DRIVING_BEHAVIOR_FEATURES:
            if feature in df.columns:
                df_ma[feature] = df[feature]
        
        # Drop any rows where key features are still missing
        all_features = available_features + [f for f in DRIVING_BEHAVIOR_FEATURES if f in df.columns]
        df_ma = df_ma.dropna(subset=all_features)
        
        # Reset index to bring timestamp back as a column if needed downstream
        df_ma.reset_index(inplace=True)
        
        # For demonstration, compute recent stats over the last window_length days
        try:
            # Check if timestamp column exists, otherwise use index
            if 'timestamp' in df_ma.columns:
                time_cutoff = df_ma['timestamp'].max() - pd.Timedelta(days=window_length)
                recent_data = df_ma[df_ma['timestamp'] > time_cutoff]
            else:
                # If no timestamp column, use the entire dataset
                print("[WARNING] No timestamp column found. Using entire dataset for recent stats.")
                recent_data = df_ma
        except Exception as e:
            print(f"[WARNING] Error calculating recent stats: {e}. Using entire dataset.")
            recent_data = df_ma
        
        # Initialize recent_stats with default values
        recent_stats = {}
        
        # Define columns to include in stats with their default values
        stats_columns = {
            "IVCU_AmbAirTemp": 0.0,
            "IBMU_StatsPkBlkVoltAvg": 0.0,
            "IBMU_StatsPkCurr": 0.0,
            "IBMU_AlgoPkSocTrue": 0.0,
            "IBMU_StatsPkTempCellMax": 0.0,
            "IBMU_StatsPkTempCellMin": 0.0,
            "IMPB_CooltTemp": 0.0,
            "IBMU_WntyTotKWhRgn": 0.0,
            "rms_current_1h": 0.0,
            "rms_current_1d": 0.0,
            "max_acceleration": 0.0,
            "avg_speed": 0.0,
            "driving_aggressiveness": 0.0,
            "battery_stress": 0.0
        }
        
        # Calculate stats for available columns
        for col, default_value in stats_columns.items():
            if col in recent_data.columns and not recent_data.empty:
                recent_stats[f"{col}_avg"] = recent_data[col].mean()
            else:
                recent_stats[f"{col}_avg"] = default_value
        
        # Prepare feature matrix X and target y
        # Target is SoH, which should not be in features
        X = df_ma[all_features].values.astype(np.float32)
        
        # Debug print to show available columns
        print(f"[DEBUG] Available columns in DataFrame: {df_ma.columns.tolist()}")
        
        # Search for any columns containing 'soh' (case-insensitive)
        soh_columns = [col for col in df_ma.columns if 'soh' in col.lower()]
        if soh_columns:
            print(f"[DEBUG] Found potential SoH columns: {soh_columns}")
        
        # Check for the target column (SoH) in different case formats
        target_column = None
        if 'ibmu_algopksoh' in df_ma.columns:
            target_column = 'ibmu_algopksoh'
            print(f"[DEBUG] Found target column: {target_column}")
        elif 'IBMU_AlgoPkSoh' in df_ma.columns:
            target_column = 'IBMU_AlgoPkSoh'
            print(f"[DEBUG] Found target column: {target_column}")
        elif 'IBMU_AlgoPkSohTrue' in df_ma.columns:
            target_column = 'IBMU_AlgoPkSohTrue'
            print(f"[DEBUG] Found target column: {target_column}")
        # If we found any SoH columns but didn't match the expected names, use the first one
        elif soh_columns:
            target_column = soh_columns[0]
            print(f"[DEBUG] Using alternative SoH column: {target_column}")
        
        if target_column is None:
            print("[WARNING] SoH target column not found. Using a default value of 100.")
            y = np.ones(len(X)) * 100.0
        else:
            print(f"[DEBUG] Using target column: {target_column}")
            y = df_ma[target_column].values.astype(np.float32)
            # Fill NaN values with 100 (assuming 100% SoH for missing values)
            y = np.nan_to_num(y, nan=100.0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        
        if is_train:
            print(f"[DEBUG] Loaded {len(X_train)} training samples, {len(X_test)} test samples.")
            return X_train, y_train, recent_stats
        else:
            print(f"[DEBUG] Loaded {len(X)} samples for evaluation.")
            return X_test, y_test, recent_stats
        
    except Exception as e:
        print(f"[ERROR] Failed to load or process data from {filepath}")
        print(f"[DEBUG] Exception: {e}")
        raise e


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

'''def check_training_eligibility(filepath, temp_threshold):
    """
    Check if a client should participate in training based on temperature conditions.
    
    Uses the 14-day mean of 'ibmu_statspktempcellmax' as a proxy for temperature.
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    last_14_days = df.sort_values(by='date', ascending=False).head(14)
    mean_temp = last_14_days['ibmu_statspktempcellmax'].mean()
    return mean_temp > temp_threshold'''


'''def data_init():
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
    
    # option 1: assign a chunk size to an environmental variable
    try:
        chunk_iter = pd.read_csv(data_path, chunksize=80000)
        df = next(chunk_iter)
        print(f"[INFO] data.py: Successfully loaded {len(df)} rows from {data_path}.")
    except Exception as e:
        print(f"[ERROR] data.py: Failed to read CSV from {data_path}")
        print(f"[DEBUG] Exception: {e}")
        sys.exit(1)'''