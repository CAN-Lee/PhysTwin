import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import subprocess
import shutil

def check_gt_and_render(pkl_path, output_video, case_name):
    print(f"Checking {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
    obj_pts = data["object_points"] # [T, N, 3]
    ctrl_pts = data["controller_points"] # [T, C, 3]
    
    T, N, _ = obj_pts.shape
    print(f"Time frames: {T}, Object points: {N}")
    if ctrl_pts is not None:
        print(f"Controller points: {ctrl_pts.shape[1]}")
    
    # Check for static points at origin (0,0,0)
    origin_threshold = 1e-5
    
    static_at_origin_obj = []
    for t in range(T):
        norms = np.linalg.norm(obj_pts[t], axis=1)
        at_origin = np.where(norms < origin_threshold)[0]
        if len(at_origin) > 0:
            static_at_origin_obj.append((t, at_origin))
    
    if static_at_origin_obj:
        print(f"Found {len(static_at_origin_obj)} frames with object points at origin.")
        # Print first few instances
        for t, idx in static_at_origin_obj[:5]:
            print(f"  Frame {t}: Indices {idx}")
    else:
        print("No object points found at origin.")

    static_at_origin_ctrl = []
    if ctrl_pts is not None:
        for t in range(T):
            norms = np.linalg.norm(ctrl_pts[t], axis=1)
            at_origin = np.where(norms < origin_threshold)[0]
            if len(at_origin) > 0:
                static_at_origin_ctrl.append((t, at_origin))
    
    if static_at_origin_ctrl:
        print(f"Found {len(static_at_origin_ctrl)} frames with controller points at origin.")
    else:
        print("No controller points found at origin.")

    # Rendering
    temp_dir = f"temp_render_{case_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    frames = []
    for t in tqdm(range(T), desc="Rendering GT Frames"):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot GT
        ax.scatter(obj_pts[t, :, 0], obj_pts[t, :, 1], obj_pts[t, :, 2], s=5, c='green', alpha=0.5, label='Ground Truth')
        
        # Plot Controller
        if ctrl_pts is not None:
            ax.scatter(ctrl_pts[t, :, 0], ctrl_pts[t, :, 1], ctrl_pts[t, :, 2], s=50, c='red', marker='x', label='Controller')
        
        # Plot Origin Point - making it large and obvious
        ax.scatter([0], [0], [0], s=200, c='black', marker='*', label='Origin (0,0,0)')
        
        # Calculate bounds from data
        all_pts = obj_pts[t]
        if ctrl_pts is not None:
            all_pts = np.concatenate([all_pts, ctrl_pts[t]], axis=0)
        
        # Add origin to bounds check
        all_pts = np.concatenate([all_pts, [[0,0,0]]], axis=0)
        
        p_min = all_pts.min(axis=0)
        p_max = all_pts.max(axis=0)
        center = (p_min + p_max) / 2.0
        max_range = np.max(p_max - p_min) / 2.0
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.set_title(f"{case_name} - GT and Controller - Frame {t}")
        ax.legend()
        
        frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
        frames.append(frame_path)

    # Synthesize Video
    ffmpeg_bin = "/usr/bin/ffmpeg"
    if not os.path.exists(ffmpeg_bin):
        ffmpeg_bin = "ffmpeg"
        
    input_pattern = os.path.abspath(os.path.join(temp_dir, 'frame_%04d.png'))
    output_abs_path = os.path.abspath(output_video)
    
    cmd = [
        ffmpeg_bin, '-y', '-loglevel', 'error', '-r', '30',
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', 
        '-pix_fmt', 'yuv420p',
        output_abs_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Video saved to {output_abs_path}")
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"FFmpeg failed: {e}")

if __name__ == "__main__":
    cases = [
        "rope_double_hand", "single_lift_rope", "single_push_rope",
        "double_lift_cloth_1", "double_lift_cloth_3", "single_clift_cloth_1",
        "single_clift_cloth_3", "single_lift_cloth", "single_lift_cloth_1"
    ]
    for case in cases:
        pkl = f"data/different_types/{case}/final_data.pkl"
        if os.path.exists(pkl):
            output = f"check_gt_{case}.mp4"
            check_gt_and_render(pkl, output, case)
        else:
            print(f"File not found: {pkl}")
