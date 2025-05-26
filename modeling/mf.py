import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import time
import os
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error
from torch.optim.lr_scheduler import CyclicLR

# from utils import root_path
root_path = '/Users/samarthinani/Git_projects/amazon_recommender'
from pathlib import Path

#  Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using GPU.")
elif torch.cuda.is_available():
    device = torch.device("cuda") # Fallback for NVIDIA GPUs if needed
    print("CUDA device found. Using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU backend found. Using CPU.")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the Matrix Factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=100):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

        # Initialize the embeddings
        self.user_embedding.weight.data.uniform_(-0.1, 0.1)
        self.item_embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)
        dot = (user_embeds * item_embeds).sum(1)
        bias = self.user_bias(user_indices).squeeze() + self.item_bias(item_indices).squeeze() + self.global_bias
        return dot + bias
    
# Define the dataset class
class RatingDataset(Dataset):
    def __init__(self, user_indices, item_indices, ratings):
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]
    
# Function to load the data
def detrend_ratings(df):
    """
    Detrend the ratings by subtracting the mean rating on a year-month basis
    """
    # Convert the 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Extract year and month from the timestamp
    df['year_month'] = df['timestamp'].dt.to_period('M')

    # Calculate the mean rating for each year-month
    mean_ratings = df[['year_month', 'rating']].groupby('year_month')['rating'].mean().reset_index()
    mean_ratings.columns = ['year_month', 'mean_rating']

    # Merge the mean ratings back to the original dataframe
    df = pd.merge(df, mean_ratings, on='year_month', how='inner')

    # Detrend the ratings
    df['detrended_rating'] = df['rating'] - df['mean_rating']

    return df[['user_id', 'parent_asin', 'detrended_rating']]   


def load_data(folder_path):
    """
    Load user-item interaction data from CSV files in the specified folder.
    """
    user_ids = []
    item_ids = []
    ratings = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")
            df = pd.read_csv(file_path)
            df = detrend_ratings(df)
            user_ids.extend(df['user_id'].values)
            item_ids.extend(df['parent_asin'].values)
            ratings.extend(df['detrended_rating'].values)

    return np.array(user_ids), np.array(item_ids), np.array(ratings) 

def get_indices(df):
    """Convert user and item IDs to indices for embedding layers."""
    uid2idx = {uid: idx for idx, uid in enumerate(df['user_id'].unique())}
    iid2idx = {iid: idx for idx, iid in enumerate(df['item_id'].unique())}
    return uid2idx, iid2idx

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, patience=3):
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_preds, train_targets = [], []
        train_loss = 0

        for user_indices, item_indices, ratings in train_loader:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            ratings = ratings.float().to(device)

            optimizer.zero_grad()
            outputs = model(user_indices, item_indices)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(ratings.cpu().numpy())

        train_rmse = root_mean_squared_error(train_targets, train_preds)

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0

        with torch.no_grad():
            for user_indices, item_indices, ratings in val_loader:
                user_indices = user_indices.to(device)
                item_indices = item_indices.to(device)
                ratings = ratings.float().to(device)

                outputs = model(user_indices, item_indices)
                loss = criterion(outputs, ratings)
                val_loss += loss.item()

                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(ratings.cpu().numpy())

        val_rmse = root_mean_squared_error(val_targets, val_preds)
        current_lr = scheduler.get_last_lr()[0]
        end_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs} - LR: {current_lr:.6f} - Train RMSE: {train_rmse:.4f} - Val RMSE: {val_rmse:.4f} - Time: {end_time - start_time:.2f}s")

        # Early Stopping Check
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return model


def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
    

batch_size = 1024
# ---------------------------------------------------------------------------
train_folder_path = os.path.join(root_path, "data", "processed", "train")
user_ids, item_ids, ratings = load_data(train_folder_path)

train_df_all = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'rating': ratings})
df = train_df_all.copy()

uid2idx, iid2idx = get_indices(df)
df['user_id'] = df['user_id'].map(uid2idx)
df['item_id'] = df['item_id'].map(iid2idx)

# Prepare the train dataset and dataloader
train_dataset = RatingDataset(df['user_id'].values, df['item_id'].values, df['rating'].values)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"Number of training samples: {len(train_dataset)}")

# ---------------------------------------------------------------------------
valid_folder_path = os.path.join(root_path, "data", "raw", "valid")
user_ids_valid, item_ids_valid, ratings_valid = load_data(valid_folder_path)

valid_df_all = pd.DataFrame({'user_id': user_ids_valid, 'item_id': item_ids_valid, 'rating': ratings_valid})

# Ensure that the validation data contains only users and items present in the training data
valid_df_all = valid_df_all.loc[valid_df_all['user_id'].isin(uid2idx.keys()) & valid_df_all['item_id'].isin(iid2idx.keys())]
valid_df_all = valid_df_all.reset_index(drop=True)

valid_df_all['user_id'] = valid_df_all['user_id'].map(uid2idx)
valid_df_all['item_id'] = valid_df_all['item_id'].map(iid2idx)
valid_df = valid_df_all.copy()

# Prepare validation dataset and dataloader
valid_dataset = RatingDataset(valid_df['user_id'].values, valid_df['item_id'].values, valid_df['rating'].values)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
print(f"Number of validation samples: {len(valid_dataset)}")

# ---------------------------------------------------------------------------

# Initialize the model, loss function, and optimizer
num_users = df.user_id.nunique()
num_items = df.item_id.nunique()
model = MatrixFactorization(num_users, num_items, embedding_dim=100).to(device)
print("model initialized with {} users and {} items.".format(num_users, num_items))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

iterations_per_epoch = len(train_loader)
step_size_val = 4 * iterations_per_epoch

scheduler = CyclicLR(
    optimizer, 
    base_lr=1e-4,      # Minimum LR
    max_lr=5e-3,       # Peak LR
    step_size_up=step_size_val,  # Number of training steps to increase LR
    mode='triangular',  # Better for stable convergence
    cycle_momentum=False  # Required with Adam
)

# load an existing model if available
model_path = "best_model.pth"
try:
    model.load_state_dict(torch.load(model_path))
    print(f"Successfully loaded weights from {model_path}")
except Exception as e:
    print(f"Error loading weights from {model_path}: {e}")
    print("Starting with a fresh model.")
    
model = model.to(device)
print("Model ready on device:", device)

# Train the model
print("Starting training...")
trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=100, patience=5)

# Save the trained model 
# print("Training complete. Saving the model...")
# model_save_path = os.path.join(root_path, "models", "mf", "mf_model.pth")
# save_model(trained_model, model_save_path)

