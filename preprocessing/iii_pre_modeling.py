# This is a script to remove null values from the data
# prepare user and item index objects and save them to the disk
# item to category mapping

import pandas as pd
import numpy as np
import os
import pickle
import sys
sys.path.append("../")
sys.path.append("../../")
from utils import root_path

def generate_entity_index_map():
    """
    Generate a mapping from item to category and save it to disk.
    df: dataframe
    """
    # Load the data
    data_folder = "data/processed/train"

    all_users = set()
    all_items = set()
    item_category_map = {}
    
    for file_name in os.listdir(os.path.join(root_path, data_folder)):
        if file_name.endswith('.csv'):
            file_path = os.path.join(root_path, data_folder, file_name)
            category = file_name.split('.')[0]  # Assuming the category is part of the filename

            print(f"Processing file: {file_name}")
            df = pd.read_csv(file_path)
            unique_items = df['parent_asin'].unique()
            
            # Create a mapping from item to category
            for item in unique_items:
                if item not in item_category_map:
                    item_category_map[item] = category

            all_items.update(unique_items)

    # Save the index maps to disk
    dump_folder = "data/processed/common_files"

    # Save the item category map to disk
    item_category_path = os.path.join(root_path, dump_folder, "item_category_map.pkl")
    with open(item_category_path, 'wb') as f:
        pickle.dump(item_category_map, f)
    print(f"Item category map saved to {item_category_path}")

if __name__ == "__main__":
    # Generate the entity index map
    generate_entity_index_map()


