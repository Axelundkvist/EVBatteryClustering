import pandas as pd

# Load the aggregated CSV file (update the file name as needed)
df = pd.read_csv("/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle1_1min.csv")

print(df.head())
print(df.columns)
print(df.info())





# Display the first few values of the column
print(df['IBMU_AlgoPkSohTrue'].head())


# Get summary statistics (count, mean, std, min, max, etc.)
print(df['IBMU_AlgoPkSohTrue'].describe())

# List unique values (if applicable)
print(df['IBMU_AlgoPkSohTrue'].unique())
