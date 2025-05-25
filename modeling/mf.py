import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import time
import os

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
    def __init__(self, num_users, num_items, embedding_dim=8):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize the embeddings
        self.user_embedding.weight.data.uniform_(-0.1, 0.1)
        self.item_embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)
        return (user_embeds * item_embeds).sum(1)  # Dot product
    
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

def load_data(folder_path):
    """
    Load user-item interaction data from CSV files in the specified folder.
    """
    user_indices = []
    item_indices = []
    ratings = []

    train_folder = os.path.join(folder_path, "train")
    for file_name in os.listdir(train_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(train_folder, file_name)
            print(f"Processing file: {file_name}")
            df = pd.read_csv(file_path)
            user_indices.extend(df['user_id'].values)
            item_indices.extend(df['parent_asin'].values)
            ratings.extend(df['rating'].values)
    
    # load mapping pickle files
    mapping_folder = os.path.join(folder_path, "matrix_factorization_files")
    with open(os.path.join(mapping_folder, "user_index_map.pkl"), 'rb') as f:
        user_index_map = pickle.load(f)

    with open(os.path.join(mapping_folder, "item_index_map.pkl"), 'rb') as f:
        item_index_map = pickle.load(f)
    
    # Convert user and item indices to their corresponding mapped indices
    user_indices = [user_index_map[user] for user in user_indices]
    item_indices = [item_index_map[item] for item in item_indices]

    return np.array(user_indices), np.array(item_indices), np.array(ratings)

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
    






def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Train the matrix factorization model.
    """
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user_indices, item_indices, ratings in train_loader:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            ratings = ratings.float().to(device)

            optimizer.zero_grad()
            outputs = model(user_indices, item_indices)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
    return model



