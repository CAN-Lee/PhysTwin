import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D

def render_gt_track(pkl_path, output_video="gt_track_viz.mp4"):
    # 1. Load Data
    print(f"Loading {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {pkl_path}")
        return
    
    # 确保是 numpy array
    if not isinstance(data, np.ndarray):
        print(f"Error: Expected numpy array, got {type(data)}")
        return
        
    print(f"Data shape: {data.shape}")
    if len(data.shape) != 3 or data.shape[2] != 3:
        print("Error: Expected shape [T, N, 3]")
        return
        
    T, N, C = data.shape
    

    # 2. Setup Video Writer
    temp_dir = "temp_gt_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 计算全局包围盒，固定视角范围
    all_pts = data.reshape(-1, 3)
    min_xyz = all_pts.min(axis=0)
    max_xyz = all_pts.max(axis=0)
    center = (min_xyz + max_xyz) / 2
    span = (max_xyz - min_xyz).max()
    if span == 0: span = 1.0
    
    # Add padding
    span *= 1.2
    
    limit_min = center - span * 0.5
    limit_max = center + span * 0.5

    print(f"Scene Bounds: {min_xyz} to {max_xyz}")

    # 3. Render Frames
    frame_paths = []
    
    print("Rendering frames...")
    for t in range(T):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 当前帧的点
        pts = data[t] # [N, 3]
        
        # 绘制点
        # 使用不同颜色区分不同 ID 的点，方便看对应关系
        colors = plt.cm.jet(np.linspace(0, 1, N))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=50, depthshade=True)
        
        # 标号
        for i in range(N):
             ax.text(pts[i, 0], pts[i, 1], pts[i, 2], str(i), fontsize=8)

        # 绘制轨迹尾迹 (Trail)
        tail_len = 10
        start_t = max(0, t - tail_len)
        for i in range(N):
            traj = data[start_t:t+1, i, :]
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=colors[i], alpha=0.5, linewidth=1)

        # 设置固定视角范围
        ax.set_xlim([limit_min[0], limit_max[0]])
        ax.set_ylim([limit_min[1], limit_max[1]])
        ax.set_zlim([limit_min[2], limit_max[2]])
        
        ax.set_title(f"GT Track: Frame {t}/{T}")
        
        # 保存帧
        f_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
        plt.savefig(f_path)
        plt.close(fig)
        frame_paths.append(f_path)
        
        if t % 20 == 0:
            print(f"Rendered frame {t}/{T}", end='\r')

    # 4. Synthesize Video
    print("\nSynthesizing video...")
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30, (width, height))
    
    for f_path in frame_paths:
        frame = cv2.imread(f_path)
        out.write(frame)
    out.release()
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Done! Saved to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, 
                        default="data/different_types/double_lift_cloth_1/gt_track_3d.pkl",
                        help="Path to pkl file")
    parser.add_argument("--out", type=str, default="gt_track_viz.mp4")
    args = parser.parse_args()
    
    render_gt_track(args.path, args.out)
