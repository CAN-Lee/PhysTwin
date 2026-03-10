import torch
import pickle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pytorch3d.ops import knn_points

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_config_for_case(case_name, config_map, default_configs):
    # 1. Check explicit mapping first
    if case_name in config_map:
        return config_map[case_name]
    
    # 2. Check heuristics
    if 'rope' in case_name.lower():
        # Heuristic for rope:
        # rope3.yaml -> single_lift_rope (Wide Grip for lifting)
        # rope2.yaml -> rope_double_hand (Balanced/Anti-Explosion)
        # rope.yaml  -> single_push_rope* (Soft, Gradient recovery)
        
        if 'single_lift_rope' in case_name:
             return default_configs['rope3']
        elif 'rope_double_hand' in case_name:
             return default_configs['rope2']
        return default_configs['rope']
    elif any(kw in case_name.lower() for kw in ['zebra', 'sloth', 'dinosor']):
        # Heuristic for softbody:
        # softbody2.yaml  -> single_lift_sloth (Aggressive, Small Radius)
        # softbody.yaml -> All others (Balanced, Large Radius)
        if 'single_lift_sloth' in case_name:
            return default_configs['softbody2']
        return default_configs['softbody']
    elif 'cloth' in case_name.lower():
        # Heuristic for cloth:
        # mpm_cloth3.yaml -> double_lift_cloth_3 (folding)
        # mpm_cloth2.yaml -> single_lift_cloth* (heavy lift)
        # mpm_cloth.yaml  -> double_lift_cloth_1, single_clift_cloth* (general)
        
        if 'double_lift_cloth_3' in case_name:
             return default_configs['cloth3']
        elif 'single_lift_cloth' in case_name:
             return default_configs['cloth2']
        else:
             return default_configs['cloth']
    
    return default_configs['default']

def check_all_connections(base_path, output_dir, configs_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Build Config Map from target_scenes
    config_map = {}
    default_configs = {}
    
    # Load specific configs to build map
    config_files = {
        'rope': 'rope.yaml',
        'rope2': 'rope2.yaml',
        'rope3': 'rope3.yaml',
        'softbody': 'softbody.yaml',
        'softbody2': 'softbody2.yaml',
        'cloth': 'mpm_cloth.yaml',
        'cloth2': 'mpm_cloth2.yaml',
        'cloth3': 'mpm_cloth3.yaml',
    }
    
    loaded_configs = {}
    
    print("Loading configurations...")
    for key, filename in config_files.items():
        path = os.path.join(configs_dir, filename)
        if os.path.exists(path):
            cfg = load_config(path)
            loaded_configs[key] = cfg
            default_configs[key] = cfg # Store as default for heuristic fallback
            
            # Map target scenes
            if 'target_scenes' in cfg and cfg['target_scenes']:
                for scene in cfg['target_scenes']:
                    config_map[scene] = cfg
            print(f"  Loaded {filename}")
        else:
            print(f"  [WARNING] Config {filename} not found.")

    default_configs['default'] = loaded_configs.get('cloth') # Fallback

    # 2. Find all cases
    cases_paths = sorted(glob.glob(os.path.join(base_path, "*")))
    valid_cases = [p for p in cases_paths if os.path.isdir(p) and os.path.exists(os.path.join(p, "final_data.pkl"))]
    
    print(f"\nFound {len(valid_cases)} cases to process.")

    for case_path in valid_cases:
        case_name = os.path.basename(case_path)
        pkl_path = os.path.join(case_path, "final_data.pkl")
        
        # 3. Determine Parameters from Config
        cfg = get_config_for_case(case_name, config_map, default_configs)
        
        if cfg is None:
            print(f"Skipping {case_name}: No matching config found.")
            continue

        radius = cfg['mpm'].get('controller_radius', 0.2)
        max_neighbors = cfg['mpm'].get('controller_max_neighbors', 16)
        stiffness = cfg['mpm'].get('controller_stiffness', 'N/A')
        
        print(f"Processing {case_name} using config (Radius: {radius}, K: {max_neighbors})...")

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
        
        ax.set_title(f"Case: {case_name}\nRadius: {radius}, K: {max_neighbors}, Stiffness: {stiffness}\nConnections: {line_count}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        output_path = os.path.join(output_dir, f"{case_name}_connection.png")
        plt.savefig(output_path)
        plt.close()
        print(f"  -> Saved to {output_path}")

if __name__ == "__main__":
    check_all_connections("data/different_types", "all_connection_checks", "configs")
