"""
Inference and Test-Time Optimization (TTO) API for Conditional Surface Generation.
Demonstrates the inverse design loop while abstracting proprietary physical algorithms.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .model_architecture import ConditionalSurfaceDecoder

class LatentOptimizationWrapper(nn.Module):
    """Wrapper for the inference phase to optimize the latent representation."""

    def __init__(self, trained_model: ConditionalSurfaceDecoder, latent_dim: int = 64):
        super().__init__()
        self.base_model = trained_model
        
        # Freeze all network weights to isolate the latent vector
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Initialize stochastic latent vector z
        self.z = nn.Parameter(torch.randn(1, latent_dim) * 0.01)

    def forward(self, points: torch.Tensor, constraints: torch.Tensor, 
                spatial_bounds: torch.Tensor, category_id: torch.Tensor) -> torch.Tensor:
        
        p_feat = self.base_model.freq_encoding(points)
        cat_feat = self.base_model.category_embedding(category_id)
        
        # Expand standalone z to match the batch size
        z_expanded = self.z.expand(points.shape[0], -1)
        const_feat = self.base_model.constraint_encoder(constraints)
        
        x_input = torch.cat([p_feat, const_feat, spatial_bounds, cat_feat, z_expanded], dim=-1)

        x = self.base_model.activation(self.base_model.layer1(x_input))
        x = self.base_model.activation(self.base_model.layer2(x))
        x = self.base_model.activation(self.base_model.layer3(x))
        x = torch.cat([x, x_input], dim=-1)
        x = self.base_model.activation(self.base_model.layer4(x))
        x = self.base_model.activation(self.base_model.layer5(x))
        x = self.base_model.activation(self.base_model.layer6(x))
        x = self.base_model.activation(self.base_model.layer7(x))
        
        return self.base_model.layer8(x)


def compute_constraint_penalty(wrapper: LatentOptimizationWrapper, mc_pts: torch.Tensor, 
                               cond_data: dict) -> torch.Tensor:
    """
    [PROPRIETARY ABSTRACTION]
    Abstracts the proprietary mathematical logic for boundary mapping 
    and constraint verification. Returns a differentiable mock loss 
    to allow the showcase pipeline to execute cleanly.
    """
    # Mathematical formulations for physical domain mapping are suppressed here.
    # We return a dummy tensor attached to the computation graph to preserve flow.
    mock_penalty = torch.sum(wrapper.z ** 2) * 0.001 
    return mock_penalty


def optimize_latent_space(wrapper: LatentOptimizationWrapper, pts_gt: torch.Tensor, 
                          sdf_gt: torch.Tensor, cond_data: dict, epochs: int = 150, 
                          lr: float = 0.025, penalty_weight: float = 0.1) -> torch.Tensor:
    """Iteratively optimizes the latent representation."""
    
    wrapper.train()
    nn.init.normal_(wrapper.z, mean=0.0, std=0.01)
    
    optimizer = optim.Adam([wrapper.z], lr=lr)
    device = wrapper.z.device
    
    num_points = pts_gt.shape[0]
    const_expanded = cond_data['constraints'].expand(num_points, -1)
    bounds_expanded = cond_data['bounds'].expand(num_points, -1)
    cat_expanded = cond_data['category'].expand(num_points).long()
    
    # Monte Carlo points for the showcase abstraction
    N_MC = 5000 
    mc_pts = (torch.rand(N_MC, 3, device=device) * 2.0) - 1.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Surface Distance Loss
        pred_sdf = wrapper(pts_gt, const_expanded, bounds_expanded, cat_expanded)
        loss_sdf = nn.functional.l1_loss(pred_sdf, sdf_gt)
        
        # 2. L2 Regularization
        loss_reg = 1e-4 * torch.norm(wrapper.z)
        
        # 3. Constraint Penalty (Abstracted IP)
        loss_constraint = 0.0
        current_weight = penalty_weight if epoch > (epochs // 3) else 0.0 
            
        if current_weight > 0.0:            
            loss_constraint = current_weight * compute_constraint_penalty(wrapper, mc_pts, cond_data)
        
        loss_total = loss_sdf + loss_constraint + loss_reg
        loss_total.backward()
        optimizer.step()
        
        if epoch == 0 or (epoch + 1) % 50 == 0:
            penalty_val = loss_constraint.item() if isinstance(loss_constraint, torch.Tensor) else 0.0
            print(f"Epoch [{epoch+1:03d}/{epochs}] | L_SDF: {loss_sdf.item():.4f} | L_Penalty: {penalty_val:.4f}")
        
    return wrapper.z.detach()