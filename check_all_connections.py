import torch
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pytorch3d.ops import knn_points

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_all_connections(base_path, output_dir, config_paths):
    os.makedirs(output_dir, exist_ok=True)
    
    # Build scene -> config mapping from target_scenes in each YAML
    scene_config_map = {}  # scene_name -> (cfg, config_filename)
    
    print("Loading configurations...")
    for config_path in config_paths:
        if not os.path.exists(config_path):
            print(f"  [WARNING] {config_path} not found, skipping.")
            continue
        cfg = load_config(config_path)
        filename = os.path.basename(config_path)
        scenes = cfg.get('target_scenes', [])
        if not scenes:
            print(f"  [WARNING] {filename} has no target_scenes, skipping.")
            continue
        tag = cfg.get('tag', os.path.splitext(filename)[0])
        for scene in scenes:
            scene_config_map[scene] = (cfg, filename, tag)
        print(f"  Loaded {filename} [tag={tag}]: {len(scenes)} scenes")
    
    print(f"\nTotal target scenes: {len(scene_config_map)}")

    processed = 0
    for case_name in sorted(scene_config_map.keys()):
        case_path = os.path.join(base_path, case_name)
        pkl_path = os.path.join(case_path, "final_data.pkl")
        
        if not os.path.exists(pkl_path):
            print(f"Skipping {case_name}: final_data.pkl not found.")
            continue
        
        cfg, cfg_file, tag = scene_config_map[case_name]

        radius = cfg['mpm'].get('controller_radius', 0.2)
        max_neighbors = cfg['mpm'].get('controller_max_neighbors', 16)
        stiffness = cfg['mpm'].get('controller_stiffness', 'N/A')
        
        print(f"Processing {case_name} [{cfg_file}] (Radius: {radius}, K: {max_neighbors})...")
        processed += 1

        # 4. Load Data
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'object_points' in data:
            obj_pts = torch.from_numpy(data['object_points'][0]).float()
        else:
            surf = torch.from_numpy(data['surface_points']).float()
            interior = torch.from_numpy(data['interior_points']).float()
            obj_pts = torch.cat([surf, interior], dim=0)
            
        ctrl_pts = torch.from_numpy(data['controller_points'][0]).float()
        
        # 5. KNN Connection
        dist, idx, _ = knn_points(ctrl_pts.unsqueeze(0), obj_pts.unsqueeze(0), K=max_neighbors)
        dist = dist.squeeze(0)
        idx = idx.squeeze(0)
        mask = dist.sqrt() < radius
        
        # 6. Visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        # Downsample object points for faster plotting if needed, but keep all for accuracy
        # Plot Object
        ax.scatter(obj_pts[:, 0], obj_pts[:, 1], s=2, c='blue', alpha=0.3, label='Object')
        # Plot Controller
        ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], s=50, c='red', marker='x', label='Controller', zorder=10)
        
        line_count = 0
        for c_idx in range(ctrl_pts.shape[0]):
            connected_p_indices = idx[c_idx][mask[c_idx]]
            if len(connected_p_indices) > 0:
                p_pos = obj_pts[connected_p_indices]
                c_pos = ctrl_pts[c_idx] # [3]
                
                # Draw lines
                # Construct line segments: (x1, y1) -> (x2, y2)
                # p_pos is [N_conn, 3]
                for p in p_pos:
                    ax.plot([c_pos[0], p[0]], [c_pos[1], p[1]], color='red', alpha=0.15, linewidth=1.0)
                    line_count += 1
        
        ax.set_title(f"Case: {case_name} [{cfg_file}]\nRadius: {radius}, K: {max_neighbors}, Stiffness: {stiffness}\nConnections: {line_count}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        sub_dir = os.path.join(output_dir, tag)
        os.makedirs(sub_dir, exist_ok=True)
        output_path = os.path.join(sub_dir, f"{case_name}_connection.png")
        plt.savefig(output_path)
        plt.close()
        print(f"  -> Saved to {output_path}")
    
    print(f"\nDone. Processed {processed}/{len(scene_config_map)} target scenes.")

if __name__ == "__main__":
    config_paths = [
        "configs/mpm_cloth.yaml",
        "configs/rope.yaml",
        "configs/softbody.yaml",
    ]
    check_all_connections("data/different_types", "all_connection_checks", config_paths)
