import pandas as pd
import matplotlib.pyplot as plt

# path to the new dataset


df_eol = pd.read_csv('/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Alaska_vin1_beginning_of_life.csv', nrows=10000)
df_mol = pd.read_csv('/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Alaska_vin1_mid_life.csv', nrows=10000)
df_bol = pd.read_csv('/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Alaska_vin1_beginning_of_life.csv', nrows=10000)

print(df_bol.info())
print(df_bol.head())
