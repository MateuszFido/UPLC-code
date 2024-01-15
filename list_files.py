import os
import re
import shutil
from pathlib import Path

def list_files(file_path):
    file_path = Path(file_path)
            
    folder_contents = os.listdir(file_path)
    
    cal_files = []   # Create empty placeholder list for calibration files
    res_files = []   # Create empty placeholder list for measurement files
    
    for file in folder_contents:
        if file.endswith(".txt"):
            match = re.search(r'(.*)mM', file)
            if match:
                cal_files.append(file)
            else:
                res_files.append(file)
        else:
            continue

    cal_files = sorted(cal_files)
    res_files = sorted(res_files)
    
    # Create paths and check if already present 
    
    path_to_cal = file_path / 'cal'
    path_to_res = file_path / 'res'
    
    if not os.path.exists(path_to_cal):
        print("Directory \"cal\" doesn't exist. Creating and copying data...")
        try:
            os.mkdir(path_to_cal)
            for cal_file in cal_files:
                shutil.copy(cal_file, path_to_cal)
        except OSError as e:
            print(f"Error {e.filename}: {e.strerror}")
    if not os.path.exists(path_to_res):
        print("Directory \"res\" doesn't exist. Creating and copying data...")
        try:
            os.mkdir(path_to_res)
            for res_file in res_files:
                shutil.copy(res_file, path_to_res)
        except OSError as e:
            print(f"Error {e.filename}: {e.strerror}")
    
    return cal_files, res_files