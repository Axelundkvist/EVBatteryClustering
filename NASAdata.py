# src/io.py
from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np


RW1File= "/Users/Axel/Documents/Master/MasterThesis/DataSets/NASA/11. Randomized Battery Usage Data Set/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW1.mat"

raw = sio.loadmat(RW1File, squeeze_me=True)   # squeeze_me = tar bort onödiga dimensioner

print(raw.keys())
cell = raw['data']                               # <class 'numpy.void'>
print(cell.dtype.names)

#('step', 'procedure', 'description')
# print(f"--------------------------------")
# print(f"cell innehåll")
# print(cell['step'])
# print(cell['procedure'])
# print(cell['description'])



''' dessa värden du vill komma åt
'Voltage_measured', 'Current_measured', 'Temperature_measured',
'Capacity', 'Time'

'''