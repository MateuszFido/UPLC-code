import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class Compound:
    def __init__(self, index, name, area, rt):
        self.index = index
        self.name = name
        self.area = area
        self.rt = rt
        
    def __str__(self):
        return f"Peak: {self.name} (index: {self.index}) with retention time: {self.rt} min, total area: {self.area} mAU."
        

def detect_and_highlight_peaks(time, baseline_corrected_data, retention_times: dict):
    
    # Find peaks in the baseline-corrected data
    peaks, prominences = find_peaks(baseline_corrected_data, prominence=1)
    compound_list = []

    '''
    # Debug
    for i in range(0, len(peaks)):                                                                       # [0] is peak indices, [1] is peak properties
        index = peaks[i]                                                                                 
        peak_area = np.trapz(baseline_corrected_data[peaks[1]['left_bases'][i]:peaks[1]['right_bases'][i]]) # integrate within each peak's boundaries
        rt = time[peaks[i]]
        if np.abs(rt - sorted(retention_times)[i]) <= 0.03:
            name = retention_times[sorted(retention_times)[i]]
        else:
            continue
        peak = Peak(index=index, name=name, area=peak_area, rt=rt)
        peak_list.append(peak)
        print(vars(peak))
    '''    
    
    for key, value in retention_times.items():
        peak_number = np.argmin(np.abs(time[peaks] - key))
        # Debug
        # print(list(time[peaks]), key, time[peaks[peak_number]])
        rt = time[peaks[peak_number]]
        rt_difference = np.abs(rt - key)
        if rt_difference <= 0.5:
            peak_contour = range(prominences['left_bases'][peak_number], prominences['right_bases'][peak_number], 1)
            area = np.round(np.trapz(baseline_corrected_data[peak_contour], peak_contour))
            index = peaks[peak_number]
            compound = Compound(index=index, name=value, area=area, rt=rt)
            print(compound)
            compound_list.append(compound)
            peaks = np.delete(peaks, peak_number)
            prominences['left_bases'] = np.delete(prominences['left_bases'], peak_number)
            prominences['right_bases'] = np.delete(prominences['right_bases'], peak_number)
        else:
            continue
    
    # Plotting chromatogram and highlighting peaks
    plt.figure(figsize=(10, 6))
    plt.plot(time, baseline_corrected_data, label='Chromatogram (Baseline Corrected)', color='blue')
    
    # Highlight peaks closest to retention times
    for i in range(len(compound_list)):
        x = time[compound_list[i].index]
        y = baseline_corrected_data[compound_list[i].index]
        plt.scatter(x, y, color='green', marker='o')
        plt.text(x*(1.01), y*(1.01), compound_list[i].name, fontsize=8)

    # Mark manually provided retention times on the plot
    plt.vlines(list(retention_times.keys()), ymin=min(baseline_corrected_data), ymax=max(baseline_corrected_data), colors='purple', linestyles='dashed', label='Theoretical retention Times')

    plt.title('Chromatogram with Detected and Closest Peaks (Baseline Corrected)')
    plt.xlabel('Time (min)')
    plt.ylabel('Absorbance (mAU)')
    plt.legend()
    plt.show()
    
    return peaks