import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt


class Peak():
    def __init__(self, index, name, area, rt):
        self.index = index
        self.name = name
        self.area = area
        self.rt = rt
        
    def __str__(self):
        return f"{self.name} with area: {self.area}."
        

def detect_and_highlight_peaks(time, baseline_corrected_data, retention_times: dict):
    
    # Find peaks in the baseline-corrected data
    peaks = find_peaks(baseline_corrected_data, prominence=5)
    peak_list = []
    '''
    for i in range(0, len(peaks[0])):                                                                       # [0] is peak indices, [1] is peak properties
        index = peaks[0][i]                                                                                 
        peak_area = np.trapz(baseline_corrected_data[peaks[1]['left_bases'][i]:peaks[1]['right_bases'][i]]) # integrate within each peak's boundaries
        rt = time[peaks[0][i]]
        if np.abs(rt - sorted(retention_times)[i]) <= 0.03:
            name = retention_times[sorted(retention_times)[i]]
        else:
            continue
        peak = Peak(index=index, name=name, area=peak_area, rt=rt)
        peak_list.append(peak)
        print(vars(peak))
    '''    
    
    for key, value in retention_times.items():
        rt = np.abs(key - next(peaks[0]))
        if rt <= 0.3: 
            
        else:
            continue
                
    # Plotting chromatogram and highlighting peaks
    plt.figure(figsize=(10, 6))
    plt.plot(time, baseline_corrected_data, label='Chromatogram (Baseline Corrected)', color='blue')
    
    # Highlight peaks closest to retention times
    plt.scatter(time[peaks[closest_peaks_indices]], closest_peaks_values, color='green', marker='o', label='Closest Peaks')

    # Mark manually provided retention times on the plot
    plt.vlines(retention_times, ymin=min(baseline_corrected_data), ymax=max(baseline_corrected_data), colors='purple', linestyles='dashed', label='Retention Times')

    plt.title('Chromatogram with Detected and Closest Peaks (Baseline Corrected)')
    plt.xlabel('Time (min)')
    plt.ylabel('Absorbance (mAU)')
    plt.legend()
    plt.show()
    
    return peaks