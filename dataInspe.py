import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the aggregated CSV file (update the file name as needed)
#df = pd.read_csv("/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle9/0e5e44212b74110600191303770541565561_5-2024.csv")
df = pd.read_csv("/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle7/2d74710400701507007811035c4774747672_5-2023.csv")
df_rms = pd.read_csv("/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/vehicle7_10secRMS.csv")
#df = pd.read_csv("/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle1_1min.csv")

print(df.head(10))
print(df_rms.head())
print(df.columns)
print(df.info())

print("--------------------------------")

print(f"IBMU_StatsPkCurr and timestamps: \n{df[['IBMU_StatsPkCurr', 'timestamp']].head(40)}")
print(f"IBMU_StatsPkCurr: \n{df['IBMU_StatsPkCurr'].describe()}")

print("--------------------------------")
# print(f"number of charging events: {((df['IBMU_StatsPkCurr'] > 0) & (df['IBMU_StatsPkCurr'].shift(1) <= 0)).sum()}")
# df['charging_event'] = (df['IBMU_StatsPkCurr'] > 0) & (df['IBMU_StatsPkCurr'].shift(1) <= 0)
# df['charge_frequency'] = df['charging_event'].rolling(window=pd.Timedelta('1d')).sum()
# print(f"charge_frequency: \n{df['charge_frequency']}") 

# Assume df has columns ['timestamp', 'IBMU_StatsPkCurr']
# and df['timestamp'] is already a datetime64 dtype.

# 0. Make sure your timestamp column is datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# 1. Flag charging moments
df['is_charging'] = df['IBMU_StatsPkCurr'] > 0

# 2. Detect the start of each charging burst
df['session_start'] = df['is_charging'] & (~df['is_charging'].shift(1, fill_value=False))

# 3. Assign a session ID by cumulatively counting starts
df['session_id'] = df['session_start'].cumsum()

# 4. Keep only charging rows and group to get raw start/end
sessions = (
    df[df['is_charging']]
      .groupby('session_id')['timestamp']
      .agg(start='first', end='last')
      .reset_index(drop=True)
)

# 5. Ensure start/end are datetime (if they weren’t already)
sessions['start'] = pd.to_datetime(sessions['start'], errors='coerce')
sessions['end']   = pd.to_datetime(sessions['end'], errors='coerce')

# 6. Now compute duration
sessions['duration'] = sessions['end'] - sessions['start']

# 7. (Optional) Filter out very short sessions
min_duration = pd.Timedelta('3min')
filtered_sessions = sessions[sessions['duration'] >= min_duration]

print(f"filtered_sessions: \n{filtered_sessions}")
print()
print(f"sessions: \n{sessions}")


# Plot histogram
# Convert to minutes as a float
sessions['duration_min'] = sessions['duration'].dt.total_seconds() / 60.0


# plt.figure()
# plt.hist(sessions['duration_min'].dropna(), bins=100)
# plt.axvline(x=3, linestyle='--')      # 3 min cut‑off
# plt.xlabel('Session Length (minutes)')
# plt.ylabel('Number of Sessions')
# plt.title('Histogram of EV Charging Session Lengths\n(3 min cut‑off shown)')
# plt.tight_layout()
# plt.show()


print("--------------------------------")

print(f"IBMU_AlgoPkSohTrue: {df['IBMU_AlgoPkSohTrue'].head()}")
# Get summary statistics (count, mean, std, min, max, etc.)
print(df['IBMU_AlgoPkSohTrue'].describe())
# List unique values (if applicable)
print(df['IBMU_AlgoPkSohTrue'].unique())

print("--------------------------------")
print(df['ibmu_algopksoh'].head())
print(df['ibmu_algopksoh'].describe())
print(df['ibmu_algopksoh'].unique())

# print("--------------------------------")
# print("--------------------------------")

# print(df['IBMU_StatsPkTempCellMax'].head())
# print(df['IBMU_StatsPkTempCellMax'].describe())
# print(df['IBMU_StatsPkTempCellMax'].unique())

# print("--------------------------------")
# print(df['IBMU_StatsPkTempCellMin'].head())
# print(df['IBMU_StatsPkTempCellMin'].describe())
# print(df['IBMU_StatsPkTempCellMin'].unique())


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
    
    
fieldVehicleDatafiles_folders=(
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle1",
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle2",
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle3",
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle4",
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle5",
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle6",
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle7",
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle8",
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle9"
)

# for folder in fieldVehicleDatafiles_folders:
#     print(f"folder: {os.listdir(folder)}")
#     print()
#     print("new folder:")
#     print(f"Folder: {folder}")
#     print(f"Number of files in folder: {len(os.listdir(folder))}")
#     for i in range(len(os.listdir(folder))):    
#         print(f"File: {pickDataFile(folder, i)}")
        
    
