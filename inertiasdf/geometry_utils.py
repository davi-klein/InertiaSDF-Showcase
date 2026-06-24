"""
Geometry utility functions for visualizing the outputs.
"""

import torch
import numpy as np
from skimage.measure import marching_cubes

def extract_marching_cubes(wrapper, cond_data: dict, grid_res: int = 64) -> tuple:
    """Samples a 3D grid and reconstructs the surface via Marching Cubes algorithm."""
    wrapper.eval()
    device = wrapper.z.device
    
    grid_range = np.linspace(-1.0, 1.0, grid_res)
    grid_x, grid_y, grid_z = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
    
    grid_pts = torch.tensor(np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1), 
                            dtype=torch.float32).to(device)
    
    base_const = cond_data['constraints'][0].unsqueeze(0)
    base_bounds = cond_data['bounds'][0].unsqueeze(0)
    base_cat = cond_data['category'][0].unsqueeze(0)
    
    sdf_values = []
    batch_size = 32768 
    
    with torch.no_grad():
        for i in range(0, grid_pts.shape[0], batch_size):
            p_chunk = grid_pts[i : i + batch_size]
            chunk_size = p_chunk.shape[0]
            
            const_chunk = base_const.expand(chunk_size, -1)
            b_chunk = base_bounds.expand(chunk_size, -1)
            cat_chunk = base_cat.expand(chunk_size)
            
            pred_sdf = wrapper(p_chunk, const_chunk, b_chunk, cat_chunk)
            sdf_values.append(pred_sdf.cpu().numpy())
            
    sdf_volume = np.concatenate(sdf_values).reshape(grid_res, grid_res, grid_res)
    
    try:
        verts, faces, _, _ = marching_cubes(sdf_volume, level=0.0)
        verts = (verts / (grid_res - 1)) * 2.0 - 1.0
        return verts, faces
    except ValueError:
        print("Warning: Marching cubes failed. The SDF might not contain a zero-level set.")
        return np.array([]), np.array([])