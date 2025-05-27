from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import os
import json
from utils import root_path


raw_data_folder = os.path.join(root_path, "data", "raw", "meta")
output_folder = os.path.join(root_path, "data", "processed", "common_files")

def parse_jsonl(file_path):
    """
    Generator to parse JSONL files line by line.
    """
    with open(file_path, 'r') as json_file:
        for line in json_file:
            yield json.loads(line)

def process_file(file_name):
    """
    Process a single JSONL file and save its metadata to a CSV file.
    """
    category_name = file_name.split('.')[0]
    meta_data = []
    file_path = os.path.join(raw_data_folder, file_name)
    for result in parse_jsonl(file_path):
        if isinstance(result, dict):
            parent_asin = result.get('parent_asin', None)
            main_category = result.get('main_category', None)
            average_rating = result.get('average_rating', None)
            rating_number = result.get('rating_count', None)

            if parent_asin:
                meta_data.append({
                    'parent_asin': parent_asin,
                    'main_category': main_category,
                    'average_rating': average_rating,
                    'rating_number': rating_number
                })

    # Create a DataFrame from the collected metadata
    df_meta = pd.DataFrame(meta_data)

    # Save the DataFrame to a CSV file
    output_path = os.path.join(output_folder, f"{category_name}.csv")
    df_meta.to_csv(output_path, index=False)
    print(f"Metadata saved to {output_path}")

def extract_meta_data():
    """
    Extract metadata from JSONL files in parallel and save them to CSV files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    jsonl_files = [file_name for file_name in os.listdir(raw_data_folder) if file_name.endswith('.jsonl')]

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, jsonl_files)

if __name__ == "__main__":
    # Extract metadata from JSONL files
    extract_meta_data()