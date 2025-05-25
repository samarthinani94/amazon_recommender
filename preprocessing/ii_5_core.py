import pandas as pd
import numpy as np
import os

def iterative_filter(df, min_user_interactions=5, min_item_interactions=3, verbose=True, max_iters=10):
    """
    Iteratively filter the dataframe to remove users and items with too few interactions.
    df: training dataframe
    min_user_interactions: minimum number of interactions per user
    min_item_interactions: minimum number of interactions per item
    verbose: whether to print the shape of the dataframe at each iteration
    max_iters: maximum number of iterations to run
    """
    prev_shape = None
    for i in range(max_iters):
        initial_shape = df.shape
        
        # Filter items with enough interactions
        item_counts = df['parent_asin'].value_counts()
        items_to_keep = item_counts[item_counts >= min_item_interactions].index
        df = df[df['parent_asin'].isin(items_to_keep)]
        
        # Filter users with enough interactions
        user_counts = df['user_id'].value_counts()
        users_to_keep = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(users_to_keep)]
        
        if verbose:
            print(f"Iteration {i+1}: shape={df.shape}, users={df['user_id'].nunique()}, items={df['parent_asin'].nunique()}")
        
        # Check if shape has converged
        if prev_shape == df.shape:
            if verbose:
                print("Converged.")
            break
        prev_shape = df.shape

        # % users with too few interactions
        user_counts = df['user_id'].value_counts()
        too_few_users = user_counts[user_counts < min_user_interactions]
        if verbose:
            print(f"Users with too few interactions: {len(too_few_users)}")

        # % items with too few interactions
        item_counts = df['parent_asin'].value_counts()
        too_few_items = item_counts[item_counts < min_item_interactions]
        if verbose:
            print(f"Items with too few interactions: {len(too_few_items)}")

    return df

if __name__ == "__main__":
    # load the data

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_folder = "data/processed/train"
    train_folder = os.path.join(project_root, data_folder)

    for file_name in os.listdir(train_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(train_folder, file_name)
            df = pd.read_csv(file_path)
            print(f"Processing {file_name}...")
            df = iterative_filter(df, min_user_interactions=5, min_item_interactions=3, verbose=True)
            df.to_csv(file_path, index=False)
            print(f"Processed data saved to {file_path}")



