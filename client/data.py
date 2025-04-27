import os
import sys
import pandas as pd
import numpy as np
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
    "fast_charging_currents",
    "DoD",
    "rms_current_1h",      # RMS current over 1-hour window
    "rms_current_1d",      # RMS current over 1-day window
    "discharge_rate",      # Rate of discharge
    "temperature_range",   # Temperature range (max-min)
]

# TODO: change the fast_charges to a parameter derived from the current parameters
# TODO: use the actual SoH parameter as the target
# TODO: find an article that acutally derives a SoH model from the data

def calculate_rms_feature(df, feature_name, window='1h'):
    """
    Calculate RMS value for a specific feature over a specified time window.
    
    Args:
        df: DataFrame with the feature column and datetime index
        feature_name: Name of the feature column to calculate RMS for
        window: Time window for calculation (default: '1h')
        
    Returns:
        Series with RMS feature values
    """
    print(f"[DEBUG] calculate_rms_feature: Calculating RMS for {feature_name} with window {window}")
    
    if feature_name not in df.columns:
        print(f"[WARNING] Column '{feature_name}' not found. Returning zeros.")
        return pd.Series(0, index=df.index)
    
    try:
        # Convert window to pd.Timedelta if it's a string
        if isinstance(window, str):
            window = pd.Timedelta(window)
            print(f"[DEBUG] calculate_rms_feature: Converted window '{window}' to Timedelta")
        
        # Calculate RMS value using rolling window
        return df[feature_name].rolling(window=window).apply(
            lambda x: np.sqrt(np.mean(x**2)) if len(x) > 0 else 0
            )

    except Exception as e:
        print(f"[WARNING] Error calculating RMS for {feature_name}: {e}. Using zeros instead.")
        print(f"[ERROR] calculate_rms_feature: Exception details: {str(e)}")
        return pd.Series(0, index=df.index)


def extract_sessions_and_features(df, n_sessions=10, fast_charge_thresh=0.5):
    """
    Extracts features and SoH target for the first n_sessions of charge and discharge.
    Returns two DataFrames: charge_df, discharge_df.
    """
    df = df.copy()
    
    # 1) Tag each row as charge, discharge, or rest
    df['process'] = np.where(df['Charge_Energy (Wh)'] > 0,
                             'charge',
                             np.where(df['Discharge_Energy (Wh)'] > 0,
                                      'discharge',
                                      'rest'))

    # 2) Assign session IDs for contiguous blocks
    df['session_id'] = (df['process'] != df['process'].shift()).cumsum()

    # 3) Group into individual sessions
    groups = df[df['process'].isin(['charge', 'discharge'])].groupby('session_id')

    # 4) Select first n_sessions of each type
    charge_grps = [(sid, g) for sid, g in groups if g['process'].iloc[0]=='charge'][:n_sessions]
    discharge_grps = [(sid, g) for sid, g in groups if g['process'].iloc[0]=='discharge'][:n_sessions]

    # 5) Extract features + SoH target for discharge
    discharge_feats = []
    for sid, g in discharge_grps:
        discharge_feats.append({
            'session_id': sid,
            'env_temp': g['Environment_Temperature (C)'].mean(),
            'rms_current': np.sqrt((g['Current (A)']**2).mean()),
            'depth_of_discharge': g['Discharge_Capacity (Ah)'].iloc[-1],
            'target_SoH': g['SoH'].iloc[-1]   # SoH at end of session
        })
    discharge_df = pd.DataFrame(discharge_feats)

    # 6) Extract features + SoH target for charge
    charge_feats = []
    for sid, g in charge_grps:
        charge_feats.append({
            'session_id': sid,
            'env_temp': g['Environment_Temperature (C)'].mean(),
            'rms_current': np.sqrt((g['Current (A)']**2).mean()),
            'fast_charge': int(g['Current (A)'].max() > fast_charge_thresh),
            'target_SoH': g['SoH'].iloc[-1]   # SoH at end of session
        })
    charge_df = pd.DataFrame(charge_feats)

    return charge_df, discharge_df


def load_data(filepath, is_train=True, window_offset=0, number_of_cycles_to_compare=10):
    """
    Load and preprocess the EV battery dataset for a specific EV client.
    
    This function:
      - Parses the 'timestamp' column to create a datetime index.
      - Sorts the data by timestamp.
      - defines what a driving session is
      - Drops rows with missing values in the key feature columns.
      - Splits data into training and testing sets.
      
    Args:
        filepath: Path to the data file
        is_train: Whether to return training or testing data
        number_of_cycles_to_compare: Length of the data window in number of cycles
    """
    print(f"[DEBUG] Loading data from {filepath}")
    # seting reminder to to rename the file to data.py later
    file_name = os.path.basename(filepath)
    print(f"[DEBUG] using {file_name} to load data")
    #print(f"[DEBUG] Using window offset: {window_offset}, window length: {window_length}")
    print(f"[DEBUG] Using window offset: {window_offset}, number_of_cycles_to_compare: {number_of_cycles_to_compare}")

    try:            
        print(f"[DEBUG] Loading data from {filepath}")
        file_name = os.path.basename(filepath)
        print(f"[DEBUG] using {file_name}")
        print(f"[DEBUG] Using window_offset={window_offset}, number_of_cycles_to_compare={number_of_cycles_to_compare}")

        
        # read the file
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, skipping.")
                
        try:
            charge_df, discharge_df = extract_sessions_and_features(df)
            #print(charge_df.head())
            #print(discharge_df.head())
            '''detta verkar fungera'''
            
            
        except: 
            print(f"ERROR in calculating sessions and their features")
        charge_df = charge_df.assign(session_type='charge',
                             # discharge-only field stays NaN in charge rows
                             depth_of_discharge=np.nan)
        discharge_df = discharge_df.assign(session_type='discharge',
                                        # charge-only field stays NaN in discharge rows
                                        fast_charge=np.nan)

        try:
            # 2) Reorder columns if you like
            cols = ['session_id', 'session_type', 'env_temp', 'rms_current',
                    'depth_of_discharge', 'fast_charge', 'target_SoH']
            charge_df = charge_df[cols]
            discharge_df = discharge_df[cols]
            #print(charge_df.head())
            #print(discharge_df.head())
        except:
            print(f"här e felet")
            
            
        # 3) Concatenate
        df = pd.concat([charge_df, discharge_df], ignore_index=True)
        
        # 2) Define your target
        y = df['target_SoH']

        # 3) Define your feature matrix
        #    Drop the columns you don’t want as predictors (target, session_id)
        X = df.drop(['target_SoH', 'session_id'], axis=1)

        # 4) Encode the session_type categorical into a dummy variable
        X = pd.get_dummies(X, columns=['session_type'], drop_first=False)
        
        #    (this will create a column named "session_type_charge" for example)

        try:
            # 4) Compute separate “recent_stats” for charge vs. discharge
            stats_charge    = X[X['session_type_charge'] == 1].mean()
            stats_discharge = X[X['session_type_charge'] == 0].mean()
            recent_stats = pd.DataFrame({
                'charge':    stats_charge,
                'discharge': stats_discharge
            })
        except Exception as e:
            print(f"ERROR in the computing recent stats: {e}")
        # 5) Now split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.1,
            random_state=42  # for reproducibility
        )
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        
        '''print(f"lite prints")
        print(f"X_train : {X_train}")
        print(f"y_train: {y_train}")
        print(f"recent_stats: {recent_stats}")
        '''
        if is_train:
            print(f"[DEBUG] Loaded {len(X_train)} training samples, {len(X_test)} test samples.")
            return X_train, y_train, recent_stats
        else:
            print(f"[DEBUG] Loaded {len(X)} samples for evaluation.")
            return X_test, y_test, recent_stats
        
    except Exception as e:
        print(f"[ERROR] Failed to load or process data from {filepath}")
        print(f"[ERROR] Exception: {e}")
        raise e


if __name__ == "__main__":
    print("[DEBUG] =================== Starting data.py 27th April 2025 ====================")
    data_path = os.getenv("FEDN_DATA_PATH")
    if data_path is None:
        #print("[ERROR] FEDN_DATA_PATH environment variable is not set!")
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

