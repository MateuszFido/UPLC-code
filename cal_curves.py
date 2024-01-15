import os
import pandas as pd
import re
import matplotlib.pyplot as plt

# Function to integrate peaks and create a calibration curve
def create_calibration_curve(calibration_files, retention_times, time, baseline_corrected_data):
    calibration_data = []

    # Iterate over all calibration files
    for calibration_file in calibration_files:
        # Load calibration data
        calibration_dataframe = load_absorbance_data(calibration_file)
        
        # Extract concentration from the filename
        concentration = extract_concentration(calibration_file)

        # Detect and highlight peaks in the calibration data
        peaks, _ = find_peaks(calibration_dataframe['Value (mAU)'], height=0)

        # Integrate peak areas
        peak_areas = np.trapz(baseline_corrected_data[peaks], dx=np.diff(time[peaks]))

        # Append data to the calibration_data list
        calibration_data.append({'Concentration (mM)': concentration, 'Peak Area': peak_areas})

    # Create a DataFrame from the calibration_data list
    calibration_df = pd.DataFrame(calibration_data)

    # Save the calibration data to a CSV file
    calibration_df.to_csv('calibration_curve.csv', index=False)

    # Plot the calibration curve
    plt.figure(figsize=(10, 6))
    plt.scatter(calibration_df['Concentration (mM)'], calibration_df['Peak Area'], label='Calibration Data', color='blue')
    plt.title('Calibration Curve')
    plt.xlabel('Concentration (mM)')
    plt.ylabel('Peak Area')
    plt.legend()
    plt.show()

# Example usage: Replace 'calibration_files' with the actual list of calibration file paths
calibration_files = [..., '0.1mM_XXXX.txt', '0.5mM_XXXX.txt', '1mM_XXXX.txt', '2.5mM_XXXX.txt', '10mM_XXXX.txt', ...]

# Call the function to create the calibration curve
create_calibration_curve(calibration_files, retention_times, time, baseline_corrected_data)
