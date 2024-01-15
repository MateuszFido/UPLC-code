import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def perform_background_correction(dataframe: pd.DataFrame, file_name):
    
    # Extract Time (min) and Value (mAU) columns
    time = dataframe.copy()['Time (min)'].values
    absorbance = dataframe.copy()['Value (mAU)'].values
    if (absorbance < 0).any():
        shift = np.median(absorbance[absorbance < 0])
    else:
        shift = 0
    absorbance -= shift
    absorbance *= np.heaviside(absorbance, 0)
    # Compute the LLS operator
    tform = np.log(np.log(np.sqrt(absorbance + 1) + 1) + 1)

    # Compute the number of iterations given the window size.
    n_iter = 20

    for i in range(1, n_iter + 1):
        tform_new = tform.copy()
        for j in range(i, len(tform) - i):
            tform_new[j] = min(tform_new[j], 0.5 *
                                (tform_new[j+i] + tform_new[j-i]))
        tform = tform_new

    # Perform the inverse of the LLS transformation and subtract
    inv_tform = ((np.exp(np.exp(tform) - 1) - 1)**2 - 1)
    baseline_corrected = np.round(
        (absorbance - inv_tform), decimals=9)
    baseline = inv_tform + shift
    
    #'''
    # Debugging
    # Plotting chromatogram before and after background correction
    plt.figure(figsize=(10, 6))

    # Before background correction
    plt.subplot(2, 1, 1)
    plt.plot(time, dataframe['Value (mAU)'].values, label='Before Correction', color='blue')
    plt.title(f'{file_name} Before Background Correction')
    plt.xlabel('Time (min)')
    plt.ylabel('Absorbance (mAU)')
    plt.legend()

    # After background correction
    plt.subplot(2, 1, 2)
    plt.plot(dataframe['Time (min)'], baseline_corrected, label='After Correction', color='green')
    plt.plot(dataframe['Time (min)'], baseline, label='Baseline', color='red', linestyle='--')
    plt.title(f'{file_name} After Background Correction')
    plt.xlabel('Time (min)')
    plt.ylabel('Absorbance (mAU)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    #'''
    
    return baseline_corrected, baseline