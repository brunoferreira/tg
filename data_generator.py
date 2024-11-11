import numpy as np
import pandas as pd

def generate_sample_data(n_users=1000, n_jobs=500):
    # Generate user features
    users = {
        'user_id': range(n_users),
        'skills': [
            np.random.choice(['Python', 'Java', 'JavaScript', 'SQL', 'C++', 'Figma'], 
                           size=np.random.randint(2, 5)) for _ in range(n_users)
        ],
        'years_experience': np.random.randint(0, 15, size=n_users),
        'location': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Salvador', 'Fortaleza', 'Rio Branco'], 
                                   size=n_users),
        'areas_interest': [
            np.random.choice(['Desenvolvimento Web', 'Data Science', 'Desenvolvimento Mobile', 'Analista de Dados', 'Designer de Produto'], 
                           size=np.random.randint(1, 4)) for _ in range(n_users)
        ]
    }
    
    # Generate job features
    jobs = {
        'job_id': range(n_jobs),
        'experience_required': np.random.randint(0, 10, size=n_jobs),
        'job_areas': [
            np.random.choice(['Desenvolvimento Web', 'Data Science', 'Desenvolvimento Mobile', 'Analista de Dados', 'Designer de Produto']) 
            for _ in range(n_jobs)
        ],
        'location': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Salvador', 'Fortaleza', 'Rio Branco'], 
                                   size=n_jobs),
        'related_skills': [
            np.random.choice(['Python', 'Java', 'JavaScript', 'SQL', 'C++', 'Figma'], 
                           size=np.random.randint(2, 4)) for _ in range(n_jobs)
        ]
    }
    
    # Generate interactions
    n_interactions = n_users * 5  # Average 5 interactions per user
    interactions = {
        'user_id': np.random.randint(0, n_users, size=n_interactions),
        'job_id': np.random.randint(0, n_jobs, size=n_interactions),
        'viewed': np.random.randint(0, 5, size=n_interactions),  # Number of views
        'applied': np.random.choice([0, 1], size=n_interactions, p=[0.8, 0.2])  # 20% apply rate
    }
    
    return pd.DataFrame(users), pd.DataFrame(jobs), pd.DataFrame(interactions) 