import torch
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.ops import knn_points

def check_zebra_connection(pkl_path, output_path):
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 1. Extract points for Frame 0
    # Note: final_data.pkl usually stores 'object_points' as [T, N, 3] 
    # or 'surface_points'/'interior_points'
    if 'object_points' in data:
        obj_pts = torch.from_numpy(data['object_points'][0]).float() # [N, 3]
    else:
        # Fallback to surface+interior
        surf = torch.from_numpy(data['surface_points']).float()
        interior = torch.from_numpy(data['interior_points']).float()
        obj_pts = torch.cat([surf, interior], dim=0)
        
    ctrl_pts = torch.from_numpy(data['controller_points'][0]).float() # [C, 3]
    
    print(f"Object points: {obj_pts.shape}, Controller points: {ctrl_pts.shape}")

    # 2. Mimic Simulator Connection Logic
    # Parameters for testing refined logic
    radius = 0.07
    max_neighbors = 64
    
    # KNN Search
    dist, idx, _ = knn_points(
        ctrl_pts.unsqueeze(0), 
        obj_pts.unsqueeze(0), 
        K=max_neighbors
    )
    
    dist = dist.squeeze(0) # [C, K]
    idx = idx.squeeze(0)   # [C, K]
    mask = dist.sqrt() < radius # [C, K]
    
    # 3. Visualization (Top View - XY Plane)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Plot Object Particles (Blue)
    ax.scatter(obj_pts[:, 0], obj_pts[:, 1], s=2, c='blue', alpha=0.3, label='Object')
    
    # Plot Controller Points (Red X)
    ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], s=50, c='red', marker='x', label='Controller', zorder=10)
    
    # Plot Connection Lines
    line_count = 0
    for c_idx in range(ctrl_pts.shape[0]):
        connected_p_indices = idx[c_idx][mask[c_idx]]
        for p_idx in connected_p_indices:
            p_pos = obj_pts[p_idx]
            c_pos = ctrl_pts[c_idx]
            ax.plot([c_pos[0], p_pos[0]], [c_pos[1], p_pos[1]], color='red', alpha=0.1, linewidth=0.5)
            line_count += 1
            
    print(f"Total connections established: {line_count}")
    
    ax.set_title(f"Top View (XY): double_lift_zebra Connection Check\nRadius: {radius}, Connections: {line_count}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Equal aspect ratio for accurate top view
    ax.set_aspect('equal', adjustable='box')
    
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    pkl = "data/different_types/double_lift_zebra/final_data.pkl"
    output = "check_zebra_top_view.png"
    if os.path.exists(pkl):
        check_zebra_connection(pkl, output)
    else:
        print(f"File not found: {pkl}")
