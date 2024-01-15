from load_data import load_absorbance_data
from bg_corr import perform_background_correction
from find_peaks import detect_and_highlight_peaks
from list_files import list_files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from collections import OrderedDict

# Log script run-time 
st = time.time()

file_path = Path('C:\\Users\\drago\\OneDrive - ETH Zurich\\Mice\\UPLC code')
RT_peaks = {1.433: 'glutamate', 
            1.886: 'int_std', 
            3.293: 'glutamine', 
            5.940: 'citrulline',
            6.406: 'X',
            7.493: 'GABA',
            7.906: 'alanine',
            10.020: 'Nh4Cl',
            12.653: 'valine',
            13.880: 'tryptophan',
            14.186: 'isoleucine',
            14.480: 'Leu-Phe',
            15.433: 'ornithine'}
RT_peaks =  dict(sorted(RT_peaks.items()))  # convert to sorted

cal_files, res_files = list_files(file_path)

print('Found calibration files:', cal_files, '\n')
print('Found measurement files:', res_files, '\n') 

# Calculate the calibration curve 
for cal_file in cal_files:
    dataframe = load_absorbance_data(file_path / cal_file)                                          # load the data into a pd.DataFrame
    baseline_corrected_data, baseline = perform_background_correction(dataframe, cal_file)          # perform background correction (bg_corr.py)
    peaks = detect_and_highlight_peaks(dataframe['Time (min)'], baseline_corrected_data, RT_peaks)  # detect peaks and align with RTs (find_peaks.py)


# Calculate the measurement files
for res_file in res_files:
    pass



# Report the run-time
et = time.time()
elapsed_time = et - st
print("Execution time: ", round(elapsed_time, 2), " seconds.")