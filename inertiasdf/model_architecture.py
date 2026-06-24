"""
Neural Network architecture models for Conditional 3D Surface Generation.

This module contains the definitions for the high-frequency coordinate 
mapping layer and the main auto-decoder network used to predict 
continuous distance fields conditioned on continuous and categorical constraints.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import weight_norm

class FrequencyEncoding(nn.Module):
    """Applies frequency-based feature mapping to input coordinates to break spectral bias."""

    def __init__(self, input_dim: int = 3, num_frequencies: int = 8, scale: float = 1.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("frequencies", frequencies * np.pi * scale)
        self.output_dim = input_dim * num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1)
        freq_expanded = self.frequencies.view(1, 1, -1)
        angles = x_expanded * freq_expanded
        out = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return out.reshape(x.shape[0], -1)


class ConditionalSurfaceDecoder(nn.Module):
    """
    Continuous surface auto-decoder with multimodal constraint projection 
    and inverted stochastic latent masking.
    """

    def __init__(self, num_shapes: int, latent_dim: int = 64, constraint_dim: int = 10, 
                 constraint_embed_dim: int = 64, spatial_bounds_dim: int = 3, category_num: int = 8, 
                 category_embed_dim: int = 64, hidden_dim: int = 512, latent_mask_prob: float = 0.20):
        super(ConditionalSurfaceDecoder, self).__init__()
        
        self.freq_encoding = FrequencyEncoding(input_dim=3, num_frequencies=8)
        self.category_embedding = nn.Embedding(category_num, category_embed_dim)
        self.latent_embedding = nn.Embedding(num_shapes, latent_dim)
        nn.init.normal_(self.latent_embedding.weight, mean=0.0, std=0.01)
        
        self.latent_mask_prob = latent_mask_prob
        
        # Feature Projection Module for continuous numerical constraints
        self.constraint_encoder = nn.Sequential(
            nn.Linear(constraint_dim, constraint_embed_dim),
            nn.Softplus(beta=20)
        )
        nn.init.xavier_uniform_(self.constraint_encoder[0].weight)
        nn.init.constant_(self.constraint_encoder[0].bias, 0.0)
        
        self.input_dim = (self.freq_encoding.output_dim + constraint_embed_dim + 
                          spatial_bounds_dim + category_embed_dim + latent_dim)
        
        self.layer1 = nn.Linear(self.input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.constant_(self.layer1.bias, 0.0)

        def create_hidden_layer(in_features, out_features):
            layer = nn.Linear(in_features, out_features)
            nn.init.xavier_uniform_(layer.weight)  
            nn.init.constant_(layer.bias, 0.0)
            return weight_norm(layer)              

        # Intermediate Backbone with Weight Normalization
        self.layer2 = create_hidden_layer(hidden_dim, hidden_dim)
        self.layer3 = create_hidden_layer(hidden_dim, hidden_dim)
        self.layer4 = create_hidden_layer(hidden_dim + self.input_dim, hidden_dim) # Skip Connection
        self.layer5 = create_hidden_layer(hidden_dim, hidden_dim)
        self.layer6 = create_hidden_layer(hidden_dim, hidden_dim)
        self.layer7 = create_hidden_layer(hidden_dim, hidden_dim)

        self.layer8 = nn.Linear(hidden_dim, 1) 
        nn.init.normal_(self.layer8.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.layer8.bias, 0.0)
        
        self.activation = nn.Softplus(beta=20)

    def forward(self, p: torch.Tensor, constraints: torch.Tensor, spatial_bounds: torch.Tensor, 
                category_id: torch.Tensor, shape_idx: torch.Tensor = None) -> torch.Tensor:
        
        p_feat = self.freq_encoding(p)
        cat_feat = self.category_embedding(category_id)
        const_feat = self.constraint_encoder(constraints)
        
        # Latent space retrieval or external injection handling
        if shape_idx is not None:
            z_feat = self.latent_embedding(shape_idx)
            if self.training and self.latent_mask_prob > 0.0:
                keep_prob = 1.0 - self.latent_mask_prob
                mask = (torch.rand(z_feat.shape[0], 1, device=z_feat.device) > self.latent_mask_prob).float()
                z_feat = (z_feat * mask) / keep_prob
        else:
            # Fallback for architectural inference without explicit index
            z_feat = torch.zeros(p.shape[0], 64, device=p.device)

        x_input = torch.cat([p_feat, const_feat, spatial_bounds, cat_feat, z_feat], dim=-1)
        
        x = self.activation(self.layer1(x_input))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = torch.cat([x, x_input], dim=-1) # Deep skip connection
        x = self.activation(self.layer4(x))
        x = self.activation(self.layer5(x))
        x = self.activation(self.layer6(x))
        x = self.activation(self.layer7(x))
        
        return self.layer8(x)