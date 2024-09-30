import os
import pandas as pd
import numpy as np
import cv2

def create_label_dictionary(data_folder, csv_file_path):
    folder_info = []

    # Loop through all subfolders in the parent folder
    for folder in os.scandir(data_folder):
        if folder.is_dir():
            # Count the number of files in the subfolder
            num_files = len([name for name in os.listdir(folder.path) if os.path.isfile(os.path.join(folder.path, name))])
            # Add information to the list
            folder_info.append({'Unicode': folder.name, 'Number': num_files})

    df = pd.DataFrame(folder_info)
    df.insert(0, 'Label', range(1, len(df) + 1))        
    # Save DataFrame to CSV file
    df.to_csv(csv_file_path, index=False)

dataset_path = "./image/dataset"
csv_path = "./data/data/label_unicode.csv"

create_label_dictionary(dataset_path, csv_path)
