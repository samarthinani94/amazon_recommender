# This is a script to remove null values from the data
# prepare user and item index objects and save them to the disk
# item to category mapping

import pandas as pd
import numpy as np
import os
import pickle

def generate_entity_index_map():
    """
    Generate a mapping from entity IDs to indices.
    df: dataframe
    entity_col: column name of the entity
    """
    # Load the data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_folder = "data/processed/train"

    all_users = set()
    all_items = set()
    item_category_map = {}
    
    for file_name in os.listdir(os.path.join(project_root, data_folder)):
        if file_name.endswith('.csv'):
            file_path = os.path.join(project_root, data_folder, file_name)
            category = file_name.split('.')[0]  # Assuming the category is part of the filename

            print(f"Processing file: {file_name}")
            df = pd.read_csv(file_path)
            unique_users = df['user_id'].unique()
            unique_items = df['parent_asin'].unique()
            
            # Create a mapping from item to category
            for item in unique_items:
                if item not in item_category_map:
                    item_category_map[item] = category


            all_users.update(unique_users)
            all_items.update(unique_items)

    # Create user and item index maps
    user_index_map = {user: idx for idx, user in enumerate(all_users)}
    item_index_map = {item: idx for idx, item in enumerate(all_items)}

    # Save the index maps to disk
    dump_folder = "data/processed/matrix_factorization_files"
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)
    user_index_path = os.path.join(dump_folder, "user_index_map.pkl")
    item_index_path = os.path.join(dump_folder, "item_index_map.pkl")

    with open(user_index_path, 'wb') as f:
        pickle.dump(user_index_map, f)
    print(f"User index map saved to {user_index_path}")
    print(f"Number of unique users: {len(user_index_map)}")

    with open(item_index_path, 'wb') as f:
        pickle.dump(item_index_map, f)
    print(f"Item index map saved to {item_index_path}")
    print(f"Number of unique items: {len(item_index_map)}")

    # Save the item category map to disk
    item_category_path = os.path.join(dump_folder, "item_category_map.pkl")
    with open(item_category_path, 'wb') as f:
        pickle.dump(item_category_map, f)
    print(f"Item category map saved to {item_category_path}")

if __name__ == "__main__":
    # Generate the entity index map
    generate_entity_index_map()


