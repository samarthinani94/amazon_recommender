# amazon_recommender
Recommender system based on Amazon Reviews dataset
## Step 1: Preprocessing the Data

The first step in preprocessing involves cleaning the raw data by removing outdated entries and duplicates. This is handled by the script `i_remove_old_data.py`. Below are the steps performed:

1. **Remove Old Data**: Any train data with a timestamp earlier than `2019-01-01` is filtered out. The timestamp is expected to be in Unix format (milliseconds).
2. **Remove Duplicates**: Duplicate rows in the dataset are removed to ensure data consistency.
3. **Save Processed Data**: The cleaned data is saved in the `data/processed/train` folder.

### How to Run the Preprocessing Script

1. Place your raw CSV files in the `data/raw/train` folder.
2. Run the script `preprocessing/i_remove_old_data.py`:
   ```bash
   python preprocessing/i_remove_old_data.py


## Step 2: 5 Core processing to training data
1. Iteratively remove all the users and items that have less than 5 ratings
2. Run the script `preprocessing/ii_5_core.py`