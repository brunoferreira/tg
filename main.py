import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from job_recommender import JobRecommender, prepare_features
from data_generator import generate_sample_data
from sklearn.model_selection import train_test_split

# Generate sample data
users_df, jobs_df, interactions_df = generate_sample_data()

# Prepare features
user_features, job_features = prepare_features(users_df, jobs_df, interactions_df)

# Split data
train_idx, test_idx = train_test_split(range(len(interactions_df)), test_size=0.2)

# Create model
model = JobRecommender(
    n_users=len(users_df),
    n_jobs=len(jobs_df)
)

# Training settings
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
n_epochs = 10
batch_size = 64

# Convert data to tensors
train_users = torch.tensor(interactions_df.iloc[train_idx]['user_id'].values)
train_jobs = torch.tensor(interactions_df.iloc[train_idx]['job_id'].values)
train_views = torch.tensor(interactions_df.iloc[train_idx]['viewed'].values > 0).float()
train_applies = torch.tensor(interactions_df.iloc[train_idx]['applied'].values).float()

# Create DataLoader for batch training
train_dataset = TensorDataset(train_users, train_jobs, train_views, train_applies)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop with progress tracking
print("Starting training...")
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    
    for batch_users, batch_jobs, batch_views, batch_applies in train_loader:
        # Get predictions
        pred = model(
            batch_users,
            batch_jobs,
            user_features[batch_users],
            job_features[batch_jobs]
        )
        
        # Calculate loss
        target = torch.stack([batch_views, batch_applies], dim=1)
        loss = criterion(pred, target)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

# Evaluation
print("\nEvaluating model...")
model.eval()

# Prepare test data
test_users = torch.tensor(interactions_df.iloc[test_idx]['user_id'].values)
test_jobs = torch.tensor(interactions_df.iloc[test_idx]['job_id'].values)
test_views = interactions_df.iloc[test_idx]['viewed'].values > 0
test_applies = interactions_df.iloc[test_idx]['applied'].values

with torch.no_grad():
    # Get predictions for test set
    test_pred = model(
        test_users,
        test_jobs,
        user_features[test_users],
        job_features[test_jobs]
    )
    
    # Convert predictions to probabilities
    view_probs = torch.sigmoid(test_pred[:, 0]).numpy()
    apply_probs = torch.sigmoid(test_pred[:, 1]).numpy()
    
    # Calculate metrics
    view_preds = (view_probs > 0.5)
    apply_preds = (apply_probs > 0.5)
    
    # View metrics
    view_auc = roc_auc_score(test_views, view_probs)
    view_precision = precision_score(test_views, view_preds)
    view_recall = recall_score(test_views, view_preds)
    
    # Apply metrics
    apply_auc = roc_auc_score(test_applies, apply_probs)
    apply_precision = precision_score(test_applies, apply_preds)
    apply_recall = recall_score(test_applies, apply_preds)

print("\nTest Results:")
print("\nView Metrics:")
print(f"AUC: {view_auc:.4f}")
print(f"Precision: {view_precision:.4f}")
print(f"Recall: {view_recall:.4f}")

print("\nApplication Metrics:")
print(f"AUC: {apply_auc:.4f}")
print(f"Precision: {apply_precision:.4f}")
print(f"Recall: {apply_recall:.4f}")

# Example recommendations for a specific user
def get_recommendations(user_id, top_k=5):
    model.eval()
    with torch.no_grad():
        # Get predictions for all jobs for this user
        user_tensor = torch.tensor([user_id] * len(jobs_df))
        job_tensor = torch.tensor(range(len(jobs_df)))
        
        predictions = model(
            user_tensor,
            job_tensor,
            user_features[user_tensor],
            job_features[job_tensor]
        )
        
        # Get application probabilities
        apply_probs = torch.sigmoid(predictions[:, 1]).numpy()
        
        # Get top k jobs
        top_jobs = np.argsort(apply_probs)[-top_k:][::-1]
        
        return [(idx, apply_probs[idx]) for idx in top_jobs]

# Show example recommendations for first user
print("\nExample Recommendations for User 0:")
recommendations = get_recommendations(0)
for job_id, prob in recommendations:
    print(f"Job {job_id}: {prob:.4f} probability of application")