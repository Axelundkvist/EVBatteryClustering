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

def calculate_rms_current(df, window='1h'):
    """
    Calculate RMS current over a specified time window.
    
    Args:
        df: DataFrame with IBMU_StatsPkCurr column and datetime index
        window: Time window for calculation (default: '1h')
        
    Returns:
        Series with RMS current values
    """
    return calculate_rms_feature(df, 'IBMU_StatsPkCurr', window)

def calculate_driving_behavior_features(df):
    """
    Calculate driving behavior features from the dataset.
    
    Args:
        df: DataFrame with datetime index and relevant columns
        
    Returns:
        DataFrame with driving behavior features
    """
    #print(f"[DEBUG] calculate_driving_behavior_features: DataFrame index type: {type(df.index)}")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_features = df.copy()
    
    # Check if we have the required columns
    has_current = 'IBMU_StatsPkCurr' in df.columns
    has_accel = 'IAccel_Long' in df.columns
    has_speed = 'IVehSpd' in df.columns
    has_voltage = 'IBMU_StatsPkVbatt' in df.columns
    has_temp_max = 'IBMU_StatsPkTempCellMax' in df.columns
    has_temp_min = 'IBMU_StatsPkTempCellMin' in df.columns
    
    if not has_current:
        print("[WARNING] Column 'IBMU_StatsPkCurr' not found. Some features will be zero.")
    
    if not has_accel:
        print("[WARNING] Column 'iaccel_long' not found. Acceleration features will be zero.")
    
    if not has_speed:
        print("[WARNING] Column 'ivehspd' not found. Speed features will be zero.")
    
    if not has_voltage:
        print("[WARNING] Column 'IBMU_StatsPkVbatt' not found. Voltage features will be zero.")
    
    if not has_temp_max or not has_temp_min:
        print("[WARNING] Temperature columns not found. Temperature features will be zero.")
    
    # Optimize: Calculate all features in a single pass if possible
    try:
        print("[DEBUG] calculate_driving_behavior_features: Calculating features efficiently")
        
        # Calculate RMS values for different features over different time windows
        if has_current:
            print("[DEBUG] calculate_driving_behavior_features: Calculating RMS current")
            df_features['rms_current_1h'] = calculate_rms_feature(df, 'IBMU_StatsPkCurr', window='1h')
            df_features['rms_current_1d'] = calculate_rms_feature(df, 'IBMU_StatsPkCurr', window='1d')
        else:
            df_features['rms_current_1h'] = 0
            df_features['rms_current_1d'] = 0
            
        if has_accel:
            print("[DEBUG] calculate_driving_behavior_features: Calculating RMS acceleration")
            df_features['rms_acceleration_1h'] = calculate_rms_feature(df, 'IAccel_Long', window='1h')
            df_features['rms_acceleration_1d'] = calculate_rms_feature(df, 'IAccel_Long', window='1d')
        else:
            df_features['rms_acceleration_1h'] = 0
            df_features['rms_acceleration_1d'] = 0
            
        if has_speed:
            print("[DEBUG] calculate_driving_behavior_features: Calculating RMS speed")
            df_features['rms_speed_1h'] = calculate_rms_feature(df, 'IVehSpd', window='1h')
            df_features['rms_speed_1d'] = calculate_rms_feature(df, 'IVehSpd', window='1d')
        else:
            df_features['rms_speed_1h'] = 0
            df_features['rms_speed_1d'] = 0
            
        if has_voltage:
            print("[DEBUG] calculate_driving_behavior_features: Calculating RMS voltage")
            df_features['rms_voltage_1h'] = calculate_rms_feature(df, 'IBMU_StatsPkVbatt', window='1h')
            df_features['rms_voltage_1d'] = calculate_rms_feature(df, 'IBMU_StatsPkVbatt', window='1d')
        else:
            df_features['rms_voltage_1h'] = 0
            df_features['rms_voltage_1d'] = 0
            
        # Calculate RMS temperature
        if has_temp_max and has_temp_min:
            print("[DEBUG] calculate_driving_behavior_features: Calculating RMS temperature")
            # Calculate average temperature first
            df['avg_temp'] = (df['IBMU_StatsPkTempCellMax'] + df['IBMU_StatsPkTempCellMin']) / 2
            df_features['rms_temp_1h'] = calculate_rms_feature(df, 'avg_temp', window='1h')
            df_features['rms_temp_1d'] = calculate_rms_feature(df, 'avg_temp', window='1d')
        else:
            df_features['rms_temp_1h'] = 0
            df_features['rms_temp_1d'] = 0
        
        # Calculate charge frequency using the session detection approach
        if has_current:
            print("[DEBUG] calculate_driving_behavior_features: Calculating charge frequency using session detection")
            
            # Create a temporary dataframe with timestamp index for session detection
            temp_df = df.copy()
            if not isinstance(temp_df.index, pd.DatetimeIndex):
                if 'timestamp' in temp_df.columns:
                    temp_df.set_index('timestamp', inplace=True)
                else:
                    print("[WARNING] No timestamp column found for session detection. Using simple approach.")
                    # Fallback to simple approach
                    df_features['charge_frequency'] = df['IBMU_StatsPkCurr'].rolling(window=pd.Timedelta('1d')).apply(
                        lambda x: np.sum(x > 0) if len(x) > 0 else 0
                    )
                    df_features['discharge_rate'] = df['IBMU_StatsPkCurr'].rolling(window=pd.Timedelta('1d')).apply(
                        lambda x: np.mean(x[x < 0]) if len(x[x < 0]) > 0 else 0
                    )
            else:
                # 1. Skapa boolean för laddning
                temp_df['is_charging'] = temp_df['IBMU_StatsPkCurr'] > 0

                # 2. Identifiera block av konsekutiva ladd‑/icke‑ladd‑tillstånd
                is_ch = temp_df['is_charging'].fillna(False)
                # Varje gång flaggan ändras startar ett nytt block
                temp_df['chg_block'] = (is_ch != is_ch.shift(1, fill_value=False)).cumsum()

                # 3. Sammanfatta varje laddblock till start/slut
                charging_sessions = (
                    temp_df[temp_df['is_charging']]    # bara rader med laddning
                    .groupby('chg_block')
                    .apply(lambda g: pd.Series({
                        'start': g.index[0],
                        'end'  : g.index[-1]
                    }))
                    .reset_index(drop=True)
                )

                # 4. Beräkna duration och filtrera bort kortare än 3 minuter
                if not charging_sessions.empty:
                    charging_sessions['duration'] = charging_sessions['end'] - charging_sessions['start']
                    filtered_sessions = charging_sessions[
                        charging_sessions['duration'] >= pd.Timedelta('3min')
                    ].copy()
                    print(f"filtered_sessions:\n{filtered_sessions}")

                    # 5. Räkna sessions per dag
                    daily_sessions = filtered_sessions.groupby(
                        filtered_sessions['start'].dt.date
                    ).size()
                    # Se till att täcka alla tidsstämplar och fyll 0 där inga sessioner
                    daily_sessions = daily_sessions.reindex(temp_df.index.date, fill_value=0)
                    df_features['charge_frequency'] = daily_sessions.reindex(
                        temp_df.index, method='ffill'
                    )
                else:
                    df_features['charge_frequency'] = 0

                # 6. Discharge rate som tidigare
                df_features['discharge_rate'] = (
                    df['IBMU_StatsPkCurr']
                    .rolling(window=pd.Timedelta('1d'))
                    .apply(lambda x: np.mean(x[x < 0]) if len(x[x < 0]) > 0 else 0)
                )


        else:
            df_features['charge_frequency'] = 0
            df_features['discharge_rate'] = 0
        
        # Calculate battery stress (RMS temperature × RMS current)
        if has_current and has_temp_max and has_temp_min:
            #print("[DEBUG] calculate_driving_behavior_features: Calculating battery stress")
            df_features['battery_stress'] = df_features['rms_temp_1h'] * df_features['rms_current_1h']
        else:
            df_features['battery_stress'] = 0
    
        # Calculate driving aggressiveness score based on RMS speed, voltage, and current
        print("[DEBUG] calculate_driving_behavior_features: Calculating driving aggressiveness")
        
        # Normalize RMS values
        if 'rms_speed_1h' in df_features.columns and 'rms_voltage_1h' in df_features.columns and 'rms_current_1h' in df_features.columns:
            # Calculate power factor (voltage × current)
            df_features['power_factor'] = df_features['rms_voltage_1h'] * df_features['rms_current_1h']
            
            # Normalize values
            norm_speed = (df_features['rms_speed_1h'] - df_features['rms_speed_1h'].min()) / \
                        (df_features['rms_speed_1h'].max() - df_features['rms_speed_1h'].min() + 1e-6)
            
            norm_power = (df_features['power_factor'] - df_features['power_factor'].min()) / \
                        (df_features['power_factor'].max() - df_features['power_factor'].min() + 1e-6)
            
            # Calculate aggressiveness score (weighted combination of speed and power)
            df_features['driving_aggressiveness'] = 0.4 * norm_speed + 0.6 * norm_power
        else:
            df_features['driving_aggressiveness'] = 0
        
    except Exception as e:
        print(f"[WARNING] Error calculating driving behavior features: {e}")
        # Initialize all features with zeros in case of error
        for feature in DRIVING_BEHAVIOR_FEATURES:
            df_features[feature] = 0
    
    # Fill NaN values with 0
    df_features = df_features.fillna(0)
    
    return df_features

def load_data(filepath, is_train=True, window_offset=0, number_of_cycles_to_compare=10):
    """
    Load and preprocess the EV battery dataset for a specific EV client.
    
    This function:
      - Loads a chunk of the CSV file.
      - Parses the 'timestamp' column to create a datetime index.
      - Sorts the data by timestamp.
      - defines what a driving session is
      - computes a sliding rms values for a window size of 10 driving cycles 
      - Calculates additional driving behavior features.
      - Drops rows with missing values in the key feature columns.
      - Splits data into training and testing sets.
      
    Args:
        filepath: Path to the data file
        is_train: Whether to return training or testing data
        window_offset: Offset for the data window (for orchestration)
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

        '''
        # ensure that the file is not too small, it should have more x amount o bites so you can acutally draw out some statistics on it
        if filepath
        '''
        

        #  ——— 1) Läs in var 5:e rad ———        
        df = pd.read_csv(filepath) # , skiprows=lambda i: i > 0 and i % 5 != 0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        print(f"tillgängliga columns är : \n{df.columns.tolist()}")
        
        # here you should define what a driving session is: 
        # simple definition to start with: 
        # start of sesssion: charging session ends and when IBMU_SatsPkCurr goes negative
        # end of session: when the IBMU_SatsPkCurr goes to 0 or positve again (indicating that the charge session is begining again)
        # the driving session can also be filtered so that it should be longer than 3min (indicating that if IBMU_SatsPkCurr fluctuates it incoporates hills where the car is selfcharging)
        # then divide the driving sessions in chunks of 10 and then calculate the rms values down below
        
        
        #  ——— 2) DETEKTERA DRIVING SESSIONS ———
        # a) Flagga discharging (körning) när ström < 0
        # adderar en ny kolonn som säger när strömmen är negativ (bilen körs)
        df['is_discharging'] = df['IBMU_StatsPkCurr'] < 0

        # b) Hitta början på varje discharging‐burst
        is_disc  = df['is_discharging'].fillna(False)
        prev1_d  = is_disc.shift(1, fill_value=False)
        df['drive_start'] = is_disc & (~prev1_d)

        # c) Tilldela varje burst ett ID
        df['drive_id'] = df['drive_start'].cumsum()

        # d) Sammanfatta varje burst till start–slut‐tider
        # ——— 2d) Grupp och ta första + sista rad för varje drive_id ———
        drive_sessions = (
            df[df['is_discharging']]
            .groupby('drive_id')
            .apply(lambda g: pd.Series({
                'start': g.index[0],   # första tids­tämplen i bursten
                'end'  : g.index[-1]   # sista tids­tämplen i bursten
            }))
            .reset_index()
        )

        # e) Beräkna varaktighet och filtrera bort < 3 min
        drive_sessions['duration'] = drive_sessions['end'] - drive_sessions['start']
        drive_sessions = drive_sessions[
            drive_sessions['duration'] >= pd.Timedelta('3min')
        ].reset_index(drop=True)
        print(f"[DEBUG] Found {len(drive_sessions)} real driving sessions ≥ 3min")


        #  ——— 3) Dela in i “fönster” av X cykler och beräkna sliding RMS ———
        # (b) Om du vill plocka ut exakt de cykler som motsvarar window_offset...:
        selected = drive_sessions.iloc[
            window_offset : window_offset + number_of_cycles_to_compare
            ].copy()
        
        try:
            for _, cycle in selected.iterrows():
                did = cycle['drive_id']
                start = cycle['start']
                end   = cycle['end']
                dur   = cycle['duration']
                # Extrahera precis de mätpunkter (rader) i df som hör till denna cykel
                cycle_rows = df[df['drive_id'] == did]
                
                try: 
                    soc_series = cycle_rows['IBMU_AlgoPkSocTrue'].dropna()
                except:
                    print(f"hittar inte soc_series")
                # 2) Beräkna diff = skillnaden rad för rad
                soc_diff = soc_series.diff().fillna(0)
                
                try:
                    # 3) Summera bara de positiva diffs (regenerering)
                    total_regen = soc_diff[soc_diff > 0].sum()
                    max_regen   = soc_diff.max()    # största enstaka hopp
                    
                    # NYTT: de fem största hoppena
                    positive_jumps = soc_diff[soc_diff > 0]
                    top5_jumps = positive_jumps.sort_values(ascending=False).head(5)
                        
                    print(f"\n=== Cycle {did} ===")
                    print(f"Start: {start}, End: {end}, Duration: {dur}")
                    print(f"Number of rows: {len(cycle_rows)}")
                    #print(cycle_rows)  # om du vill se all info, eller t.ex. cycle_rows.head() för bara de första
                
                    print(f"Start SoC: {soc_series.iloc[0]:.2f}%")
                    print(f"End   SoC: {soc_series.iloc[-1]:.2f}%")
                    print(f"ΔSoC   : {soc_series.iloc[-1] - soc_series.iloc[0]:.2f}%")
                
                
                    #print(f"Start SoC: {soc_series.iloc[0]:.1f}%  End SoC: {soc_series.iloc[-1]:.1f}%  ΔSoC: {soc_series.iloc[-1] - soc_series.iloc[0]:.1f}%")
                    if total_regen > 0:
                        print(f"  → Totalt SoC-hopp under cykeln: +{total_regen:.2f}% (maxhopp: +{max_regen:.2f}%)")
                    else:
                        print("  → Ingen positiv SoC-ökning (regenerering) under cykeln")
                        
                    if not top5_jumps.empty:
                        print("  → Topp 5 SoC-hopp (värde @ tidsstämpel):")
                        for ts, jump in top5_jumps.items():
                            print(f"     +{jump:.2f}% @ {ts}")
                    else:
                        print("  → Inga positiva SoC-hopp att lista.")

                
                except:
                    print(f"soc beräkningen gick inte hem")
                
                # 4) Om du vill se hela serien:
                #print(soc_series)
            
                selected = drive_sessions.iloc[
                    window_offset : window_offset + number_of_cycles_to_compare
                ].copy()
            print(f"[DEBUG] Selected cycles {window_offset}–{window_offset+number_of_cycles_to_compare-1}:")
            print(selected[['start','end','duration']])

                
        except:
            print("debuggar för att se till att selected cycles lyckas")

        # ——— 4) Fortsätt med övriga features och split ———
        y = df['ibmu_algopksoh']
        print(f"[DEBUG] df rows: {len(df)}, y entries: {len(y)}")
        
        # # 1) get all distinct calendar‐days in order
        # all_days = df.index.normalize().unique()
        # print(f"[DEBUG] {len(all_days)} unique calendar days: {all_days}")

        # # 2) compute the slice [offset : offset+length]
        # selected_days = all_days[window_offset : window_offset + window_length]
        # # 3) mask original rows by whether their date is in that slice
        # mask = df.index.normalize().isin(selected_days)
        # df = df.loc[mask]
        
        # print(f"[DEBUG] Selected data window: {selected_days[0].date()} to {selected_days[-1].date()}")
            
        # y = df['ibmu_algopksoh']
        
        ''' det som är utkommenterat var det du hade med window length'''
        ''' nu undersöker du om det finns artiklar som förklarar sitt dataflöde vid federated learning'''
        ''' just nu får du beskriva detta dataflödet som ett sätt som var på önskan av din SUpervisor '''
        
        
        # print(f"[DEBUG] df rows: {df.shape[0]}, y entries: {y.shape[0]}")
        
        print(f"[DEBUG] indices aligned? {df.index.equals(y.index)}")

        
        # Calculate additional driving behavior features
        #print("[DEBUG] Calculating driving behavior features...")
        #df = calculate_driving_behavior_features(df)
        df = calculate_driving_behavior_features(selected)
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
        
        
        # dehär ska du ta bort
        '''try:
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
            print(f"[ERROR] Error in rolling window calculation: {e}. Using simple mean instead.")
            # Fallback to simple mean if rolling fails
            df_ma = df[available_features].mean().to_frame().T
            # Repeat the mean for each row in the original dataframe
            df_ma = pd.concat([df_ma] * len(df), ignore_index=True)
            df_ma.index = df.index'''
        
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
        # här ska du inkludera recent stats för de 10 valda körningscyklarna   
        try: # här ska du inkludera recent stats för de 10 valda körningscyklarna  
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
        
        # Print all column names in lowercase for easier comparison
        print(f"[DEBUG] All column names in lowercase: {[col.lower() for col in df_ma.columns]}")
        
        # Search for any columns containing 'soh' (case-insensitive)
        # soh_columns = [col for col in df_ma.columns if 'soh' in col.lower()]
        # if soh_columns:
        #     print(f"[DEBUG] Found potential SoH columns: {soh_columns}")
        # else:
        #     print("[DEBUG] No columns containing 'soh' were found")
            
        # Also check for columns containing 'health' or 'battery'
        # health_columns = [col for col in df_ma.columns if 'health' in col.lower() or 'battery' in col.lower()]
        # if health_columns:
        #     print(f"[DEBUG] Found potential health/battery columns: {health_columns}")
        
        # # Check specifically for the 'ibmu_algopksoh' column
        # target_column = 'ibmu_algopksoh'
        
        # if target_column not in df_ma.columns:
        #     print(f"[WARNING] Target column '{target_column}' not found. Using a default value of 100.")
        #     y = np.ones(len(X)) * 100.0
        # else:
        #     print(f"[DEBUG] Using target column: {target_column}")
        #     y = df_ma[target_column].values.astype(np.float32)
        #     # Fill NaN values with 100 (assuming 100% SoH for missing values)
        #     y = np.nan_to_num(y, nan=100.0)
        
        print(f"[DEBUG] y.shape: {y.shape}")
        print(f"[DEBUG] X.shape: {X.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        
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
    print("[DEBUG] =================== Starting data.py 17th April 2025 ====================")
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

