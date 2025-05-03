import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set up paths
(dir_path, abs_path) = (os.path.dirname(os.path.realpath(__file__)), os.path.abspath(os.path.dirname(os.path.realpath(__file__))))

# Define columns present in the new dataset
desired_columns = [
    "index", "charge", "discharge", "soh", "time",
    "v_measured", "c_measured", "temp", "v_load",
    "c_load", "capacity", "ambient_temp"
]

# Driving behavior features (optional additional engineering)
DRIVING_BEHAVIOR_FEATURES = [
    "fast_charging_currents",
    "DoD",
    "rms_current_1h",      # RMS current over 1-hour window
    "rms_current_1d",      # RMS current over 1-day window
    "discharge_rate",      # Rate of discharge
    "temperature_range",   # Temperature range (max-min)
]

def calculate_rms_feature(df, feature_name, window='1h'):
    """
    Calculates rolling RMS of a given feature over a specified window.
    """
    try:
        return df[feature_name].pow(2).rolling(window).mean().pow(0.5)
    except Exception as e:
        print(f"[ERROR] calculate_rms_feature: {e}")
        return pd.Series(0, index=df.index)


def extract_sessions_and_features(df, n_sessions=10, fast_charge_thresh=0.5):
    """
    Extracts session-level features and SoH targets for the first n_sessions
    of charging and discharging cycles.
    Returns two DataFrames: charge_df and discharge_df.
    """
    df = df.copy()
    # 1) Tag each row as charge, discharge, or rest
    df['process'] = np.where(
        df['charge'] > 0,
        'charge',
        np.where(df['discharge'] > 0, 'discharge', 'rest')
    )
    # 2) Assign session IDs for contiguous blocks
    df['session_id'] = (df['process'] != df['process'].shift()).cumsum()
    # 3) Group into sessions and select first n_sessions of each type
    groups = df[df['process'].isin(['charge', 'discharge'])].groupby('session_id')
    charge_grps    = [(sid, g) for sid, g in groups if g['process'].iloc[0]=='charge'][:n_sessions]
    discharge_grps = [(sid, g) for sid, g in groups if g['process'].iloc[0]=='discharge'][:n_sessions]

    # 4) Extract discharge features
    discharge_feats = []
    for sid, g in discharge_grps:
        discharge_feats.append({
            'session_id': sid,
            'env_temp':       g['ambient_temp'].mean(),
            'mean_pack_temp': g['temp'].mean(),       # added mean temp
            'var_pack_temp':  g['temp'].var(),        # added temp variance
            'rms_current':    np.sqrt((g['c_measured']**2).mean()),
            'depth_of_discharge': g['discharge'].iloc[-1],
            'target_SoH':     g['soh'].iloc[-1]
        })
    discharge_df = pd.DataFrame(discharge_feats)

    # 5) Extract charge features
    charge_feats = []
    for sid, g in charge_grps:
        charge_feats.append({
            'session_id': sid,
            'env_temp':       g['ambient_temp'].mean(),
            'mean_pack_temp': g['temp'].mean(),       # added mean temp
            'var_pack_temp':  g['temp'].var(),        # added temp variance
            'rms_current':    np.sqrt((g['c_measured']**2).mean()),
            'fast_charge':    int(g['c_measured'].max() > fast_charge_thresh),
            'target_SoH':     g['soh'].iloc[-1]
        })
    charge_df = pd.DataFrame(charge_feats)

    return charge_df, discharge_df


def load_data(filepath, is_train=True, window_offset=0, number_of_cycles_to_compare=10):
    """
    Load and preprocess the EV battery dataset with the new schema.
    Returns (X_train/X_test, y_train/y_test, recent_stats).
    """
    print(f"[DEBUG] Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)
    # Keep only desired columns
    df = df[desired_columns]
    # Parse timestamp and sort
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df.dropna(subset=desired_columns)

    # Extract sessions
    try:
        charge_df, discharge_df = extract_sessions_and_features(
            df, n_sessions=number_of_cycles_to_compare
        )
    except Exception as e:
        print(f"[ERROR] extracting sessions: {e}")
        sys.exit(1)

    # Label session types
    charge_df    = charge_df.assign(session_type='charge')
    discharge_df = discharge_df.assign(session_type='discharge')

    # ---- Forecast future SoH using linear trend ----
    # Use discharge sessions to model SoH degradation
    cycles = np.arange(len(discharge_df)).reshape(-1, 1)
    soh_values = discharge_df['target_SoH'].values
    lr_model = LinearRegression().fit(cycles, soh_values)
    future_cycle = len(discharge_df) + 10  # 10 cycles ahead
    forecast_SoH_10 = lr_model.predict(np.array([[future_cycle]]))[0]

    # Combine for model input
    data = pd.concat([charge_df, discharge_df], ignore_index=True)
    y = data['target_SoH']
    X = data.drop(['target_SoH', 'session_id'], axis=1)
    X = pd.get_dummies(X, columns=['session_type'], drop_first=False)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # ---- Compute recent_stats on training data ----
    feature_means = X_train.mean()
    feature_vars  = X_train.var()
    soh_mean      = y_train.mean()
    soh_var       = y_train.var()
    stats_df = pd.DataFrame({'mean': feature_means, 'var': feature_vars})
    stats_df.loc['SoH'] = [soh_mean, soh_var]

    recent_stats = {
        'feature_stats': stats_df,
        'forecast_SoH_10': forecast_SoH_10
    }

    if is_train:
        return X_train, y_train, recent_stats
    else:
        return X_test, y_test, recent_stats


if __name__ == "__main__":
    data_path = os.getenv("FEDN_DATA_PATH")
    if not data_path or not os.path.exists(data_path):
        print(f"[ERROR] FEDN_DATA_PATH not set or file not found: {data_path}")
        sys.exit(1)
    X, y, stats = load_data(data_path)
    print(f"[DEBUG] X shape: {X.shape}, y length: {len(y)}")
    print(f"[DEBUG] Recent stats (features + SoH):\n{stats['feature_stats']}")
    print(f"[DEBUG] Forecasted SoH after 10 cycles: {stats['forecast_SoH_10']}")
