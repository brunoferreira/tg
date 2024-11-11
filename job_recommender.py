import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class JobRecommender(nn.Module):
    def __init__(self, n_users, n_jobs, n_factors=50, hidden_size=100):
        super(JobRecommender, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.job_embedding = nn.Embedding(n_jobs, n_factors)
        
        # Calculate input sizes
        user_feature_size = n_factors + 13  # embedding(50) + experience(1) + location(5) + skills(7)
        job_feature_size = n_factors + 13   # embedding(50) + experience(1) + location(5) + skills(7)
        
        # Feature processing layers
        self.user_features = nn.Sequential(
            nn.Linear(user_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.job_features = nn.Sequential(
            nn.Linear(job_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2)  # 2 outputs: view_score, apply_score
        )
    
    def forward(self, user_ids, job_ids, user_features, job_features):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        job_emb = self.job_embedding(job_ids)
        
        # Concatenate embeddings with features
        user_combined = torch.cat([user_emb, user_features], dim=1)
        job_combined = torch.cat([job_emb, job_features], dim=1)
        
        # Process features
        user_hidden = self.user_features(user_combined)
        job_hidden = self.job_features(job_combined)
        
        # Combine and get final prediction
        combined = torch.cat([user_hidden, job_hidden], dim=1)
        output = self.final_layers(combined)
        
        return output

# Helper function to prepare features
def prepare_features(users_df, jobs_df, interactions_df):
    # Encode categorical features
    location_encoder = LabelEncoder()
    all_locations = pd.concat([users_df['location'], jobs_df['location']]).unique()
    location_encoder.fit(all_locations)
    
    # Create one-hot encodings
    user_locations = torch.zeros((len(users_df), 5))
    user_locations[torch.arange(len(users_df)), 
                  torch.tensor(location_encoder.transform(users_df['location']))] = 1
    
    job_locations = torch.zeros((len(jobs_df), 5))
    job_locations[torch.arange(len(jobs_df)), 
                 torch.tensor(location_encoder.transform(jobs_df['location']))] = 1
    
    # Create skill indicators (simplified)
    skill_list = ['Python', 'Java', 'JavaScript', 'SQL', 'C++', 'React', 'DevOps']
    user_skills = torch.tensor([[1 if skill in user_skills else 0 for skill in skill_list] 
                               for user_skills in users_df['skills']])
    job_skills = torch.tensor([[1 if skill in job_skills else 0 for skill in skill_list] 
                              for job_skills in jobs_df['related_skills']])
    
    # Combine features
    user_features = torch.cat([
        torch.tensor(users_df['years_experience']).float().unsqueeze(1),
        user_locations,
        user_skills
    ], dim=1)
    
    job_features = torch.cat([
        torch.tensor(jobs_df['experience_required']).float().unsqueeze(1),
        job_locations,
        job_skills
    ], dim=1)
    
    return user_features, job_features 