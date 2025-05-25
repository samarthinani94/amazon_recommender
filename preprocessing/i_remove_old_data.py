import pandas as pd
import numpy as np
import os

# Define the preprocessing pipeline

def preprocess_data(raw_folder, processed_folder, file_name):
    """
    Remove any data that is before 2019-01-01
    timestamp col is unix timestamp e.g. 1545114749534
    """
    raw_path = os.path.join(raw_folder, file_name)
    processed_path = os.path.join(processed_folder, file_name)

    df = get_data(raw_path)
    df = remove_old_data(df)
    df = remove_duplicates(df)
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

def get_data(file_path):
    """
    Get the data from the CSV file
    """
    df = pd.read_csv(file_path)
    return df

def remove_old_data(df):
    """
    Remove any data that is before 2019-01-01
    timestamp col is unix timestamp e.g. 1545114749534
    """
    unix_timestamp_20190101 = pd.to_datetime('2019-01-01').timestamp() * 1000
    df = df[df['timestamp'] >= unix_timestamp_20190101]
    return df

def remove_duplicates(df):
    """
    Remove duplicates from the dataframe
    """
    df = df.drop_duplicates()
    return df


# for all file in raw/train folder 
def process_all_files(raw_folder, processed_folder):
    """
    Process all files in the raw folder
    """
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    for file_name in os.listdir(raw_folder):
        if file_name.endswith('.csv'):
            preprocess_data(raw_folder, processed_folder, file_name)

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the raw and processed folders relative to the project root
    project_root = os.path.dirname(script_dir)
    raw_folder = os.path.join(project_root, 'data/raw/train')
    processed_folder = os.path.join(project_root, 'data/processed/train')

    # Process all files in the raw folder
    process_all_files(raw_folder, processed_folder)