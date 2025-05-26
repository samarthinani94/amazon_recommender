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

    train_folder = os.path.join(folder_path, "train")
    for file_name in os.listdir(train_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(train_folder, file_name)
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
    
    
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Train the matrix factorization model.
    """
    for epoch in tqdm(range(num_epochs)):
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

def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
    
    
    
folder_path = os.path.join(root_path, "data", "processed")
user_ids, item_ids, ratings = load_data(folder_path)

# Detrend the ratings
df_all = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'rating': ratings})
df = df_all.copy()

uid2idx, iid2idx = get_indices(df)
df['user_id'] = df['user_id'].map(uid2idx)
df['item_id'] = df['item_id'].map(iid2idx)

# Prepare the dataset and dataloader
dataset = RatingDataset(df['user_id'].values, df['item_id'].values, df['rating'].values)
train_loader = DataLoader(dataset, batch_size=8192, shuffle=True)
print(f"Number of training samples: {len(dataset)}")

# Initialize the model, loss function, and optimizer
num_users = df.user_id.nunique()
num_items = df.item_id.nunique()
model = MatrixFactorization(num_users, num_items, embedding_dim=100).to(device)
print("model initialized with {} users and {} items.".format(num_users, num_items))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Train the model
print("Starting training...")
trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=5)

# Save the trained model 
print("Training complete. Saving the model...")
model_save_path = os.path.join(root_path, "models", "mf", "mf_model.pth")
save_model(trained_model, model_save_path)

