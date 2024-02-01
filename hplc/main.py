from bg_corr import perform_background_correction
from cal_curves import calibrate
from find_peaks import detect_and_highlight_peaks
from list_files import read_files
from calc_conc import calculate_concentration
from load_data import load_absorbance_data
import time
from pathlib import Path
import csv 
import re

def run():
    # Log script run-time 
    st = time.time()

    file_path = Path('/Users/mateuszfido/Library/CloudStorage/OneDrive-ETHZurich/Mice/UPLC code')
    RT_peaks = {1.433: 'Glu', 
                1.886: 'internal standard', 
                3.293: 'Gln', 
                5.940: 'Citrulline',
                6.406: 'X',
                7.493: 'GABA',
                7.906: 'Ala',
                10.020: 'NH4Cl',
                12.653: 'Val',
                13.880: 'Trp',
                14.186: 'Ile',
                14.480: 'Leu-Phe',
                15.433: 'Orn'}
    RT_peaks =  dict(sorted(RT_peaks.items()))  # convert to sorted

    cal_files, res_files = read_files(file_path)

    print('Found calibration files:', cal_files, '\n')
    print('Found measurement files:', res_files, '\n') 

    all_compounds = []

    def extract_concentration(filename):
        match = re.search(r'(\d+(\.\d+)?)mM', filename)
        if match:
            return float(match.group(1))
        else:
            return None

    # Calculate the calibration curves
    for cal_file in cal_files:
        print(f'Analyzing {cal_file}...')
        # Step 1: Load the .txt data into a pd.DataFrame (load_absorbance_data.py)
        dataframe = load_absorbance_data(file_path / cal_file)
        # Step 2: Perform background correction (bg_corr.py)
        baseline_corrected_data, baseline = perform_background_correction(dataframe, cal_file, file_path)
        # Step 3: Detect peaks and align with theoretical RTs (find_peaks.py)  
        compounds = detect_and_highlight_peaks(dataframe['Time (min)'], baseline_corrected_data, RT_peaks, cal_file, file_path)
        # Step 4: Assign the concentration values from the current file to the compounds  
        concentration = extract_concentration(cal_file)
        for compound in compounds:
            compound.concentration = concentration
        all_compounds.extend(compounds)
        
    # Step 5: Graph and return the curve parameters 
    curve_params = calibrate(all_compounds, file_path)
    print(curve_params)
    concentrations = {}

    # Repeat the same process for the measurement files
    for res_file in res_files:
        print(f'Analyzing {res_file}...')
        dataframe = load_absorbance_data(file_path / res_file)
        baseline_corrected_data, baseline = perform_background_correction(dataframe, res_file, file_path)
        compounds = detect_and_highlight_peaks(dataframe['Time (min)'], baseline_corrected_data, RT_peaks, res_file, file_path)
        compounds_in_file = {}
        for compound in compounds:
            concentration = calculate_concentration(compound.area, curve_params[compound.name])
            print(f"Compound {compound.name} in file {res_file} has a concentration of {concentration} mM." )
            compounds_in_file[compound.name] = [f'{concentration} mM', 
                                                curve_params[compound.name]['Slope'], 
                                                curve_params[compound.name]['Intercept'],
                                                curve_params[compound.name]['R-squared']]
        concentrations[res_file] = compounds_in_file
        
        with open(file_path / 'results.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write header row
            header_row = ['File']
            writer.writerow(header_row)

            # Write data rows
            for outer_key, inner_dict in concentrations.items():
                writer.writerow([outer_key, 'Compound', 'Concentration', 'Slope', 'Intercept', 'R-squared'])
                for compound_name, compound_values in inner_dict.items():
                    data_row = ['', compound_name] + compound_values
                    writer.writerow(data_row)
                
                writer.writerow([])

    # Report the run-time
    et = time.time()
    elapsed_time = et - st
    print("Execution time: ", round(elapsed_time, 2), " seconds.")
    
run()