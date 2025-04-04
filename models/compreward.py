import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Fake Label Generator in CompReward"""
    
    def __init__(self, feature_dim=384):
        super(Generator, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = F.relu(x)
        return x

class Rewarder(nn.Module):
    """Pseudo Label Reward in CompReward"""

    def __init__(self, label_dim, label_embedding_dim, feature_dim=384):
        super(Rewarder, self).__init__()

        # Feature Processing Part
        self.feature_fc = nn.Linear(feature_dim, 128)
        self.feature_norm = nn.LayerNorm(128)
        
        # Label Embedding Part
        self.label_embedding = nn.Embedding(label_dim, label_embedding_dim)
        self.label_norm = nn.LayerNorm(label_embedding_dim)
        
        # Cross-Attention Mechanism
        self.cross_attention_fc = nn.Linear(128, 1)
        
        # MLP (Multi-Layer Perceptron)
        self.mlp_fc1 = nn.Linear(128, 256)
        self.mlp_fc2 = nn.Linear(256, 128)
        
        # Feed-Forward Network (FFN)
        self.ffn_fc1 = nn.Linear(128, 64)
        self.ffn_fc2 = nn.Linear(64, 1)

    def forward(self, features, label_indices):
        # Process Features
        features = self.feature_fc(features)
        features = self.feature_norm(features)
        # Process Labels
        label_embed = self.label_embedding(label_indices)
        label_embed = self.label_norm(label_embed)
        # Cross-Attention Mechanism
        cross_attention_input = torch.cat((features, label_embed), dim=0)
        cross_attention_weights = torch.softmax(self.cross_attention_fc(cross_attention_input), dim=0)
        cross_attention_output = (cross_attention_weights * cross_attention_input).sum(dim=0)
        
        # MLP Part
        mlp_input = torch.add(cross_attention_output.unsqueeze(0).expand(label_embed.size(0), -1), label_embed)
        mlp_output = F.relu(self.mlp_fc1(mlp_input))
        mlp_output = self.mlp_fc2(mlp_output)

        # FFN Part
        ffn_output = F.relu(self.ffn_fc1(mlp_output))
        reward = torch.sigmoid(self.ffn_fc2(ffn_output))
        return reward

def cosine_similarity_n(x, y):

    # Calculate cosine similarity along the last dimension (dim=-1)
    cosine_similarity = torch.cosine_similarity(x, y, dim=-1, eps=1e-8)

    # Reshape the result to [first size of x, 1]
    normalized_similarity = (cosine_similarity + 1) / 2
    normalized_similarity = normalized_similarity.view(x.size(0), 1)

    return normalized_similarity

def add_gaussian_noise(tensor, mean=0, std=1):
    noise = torch.randn_like(tensor) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor

def label_dim(x, default_dim=100):
    return int(max(default_dim, x))