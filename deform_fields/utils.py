import torch
import numpy as np
import os
import subprocess
import shutil
import matplotlib
matplotlib.use('Agg') # Headless
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pytorch3d.ops import corresponding_points_alignment, knn_points
from pytorch3d.transforms import matrix_to_quaternion

def compute_se3_clusters(tracks, num_clusters, device='cpu', v_canonical=None):
    """
    Compute SE(3) motion primitives via clustering and Procrustes alignment.
    
    tracks: (T, N, 3) - 3D trajectories of points
    num_clusters: K
    device: torch device
    v_canonical: (M, 3) - Optional canonical vertices for weight assignment
    
    Returns:
    init_rot: (T, K, 4) - Initial rotations as quaternions (w, x, y, z)
    init_trans: (T, K, 3) - Initial translations
    init_weights: (M, K) - Initial skinning weights
    """
    T, N, _ = tracks.shape
    # Flatten trajectories for clustering: (N, T*3)
    trajectories = tracks.transpose(0, 1).reshape(N, -1).cpu().numpy()
    
    print(f"  Clustering {N} tracks into {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(trajectories)
    labels = kmeans.labels_ # (N,)
    
    # Initialize SE(3) fields
    init_rot = torch.zeros(T, num_clusters, 4)
    init_rot[..., 0] = 1.0 # w=1
    init_trans = torch.zeros(T, num_clusters, 3)
    
    # Convert tracks to torch for alignment
    tracks_torch = tracks.to(device).float()
    
    # By default, we use the first frame of tracks as the canonical source for registration
    # if no external v_canonical is provided.
    tracks_canonical = tracks_torch[0]
    
    for k in range(num_clusters):
        # Indices of points in cluster k
        idx = np.where(labels == k)[0]
        if len(idx) < 3: # Need at least 3 points for rigid alignment
            if len(idx) > 0:
                for t in range(T):
                    init_trans[t, k] = torch.mean(tracks_torch[t, idx] - tracks_canonical[idx], dim=0)
            continue
            
        p_can = tracks_canonical[idx].unsqueeze(0) # (1, M, 3)
        
        for t in range(T):
            pt = tracks_torch[t, idx].unsqueeze(0) # (1, M, 3)
            
            # Rigid alignment (Orthogonal Procrustes) from p_can to pt
            # Finds R, T such that pt = p_can * R + T (where R is right-multiplication)
            res = corresponding_points_alignment(p_can, pt, estimate_scale=False)
            
            R_right = res.R[0] # (3, 3)
            t_vec = res.T[0] # (3)
            
            # PyTorch3D's quaternion_to_matrix returns a left-multiplication matrix R_left.
            # Since R_right = R_left^T, we convert R_right^T to quaternion.
            R_left = R_right.transpose(-1, -2)
            q = matrix_to_quaternion(R_left)
            
            init_rot[t, k] = q
            init_trans[t, k] = t_vec
            
    # Initial weights: Softmax based on distance to tracks in canonical frame
    M = v_canonical.shape[0] if v_canonical is not None else N
    
    print(f"  Computing soft cluster weights for {M} vertices via distance to tracks...")
    # tracks_canonical: (N, 3), v_canonical: (M, 3)
    # Using KNN to find nearest track points and their labels
    from pytorch3d.ops import knn_points
    
    # K=5 for smoother weight distribution
    dist, idx, _ = knn_points(v_canonical.unsqueeze(0).to(device), tracks_canonical.unsqueeze(0), K=5)
    
    # dist shape: (1, M, K)
    # idx shape: (1, M, K)
    dist = dist.squeeze(0)
    idx = idx.squeeze(0)
    
    # Map indices to labels
    # labels is (N,) numpy array
    neighbor_labels = torch.from_numpy(labels).to(device)[idx].long() # (M, K)
    
    # Compute soft weights
    # weights = exp(-dist / sigma)
    sigma = 0.01 # Adjust based on scene scale
    exp_dist = torch.exp(-dist / sigma)
    
    init_weights = torch.zeros(M, num_clusters, device=device)
    for k in range(5):
        # Scatter the weights based on neighbor labels
        init_weights.scatter_add_(1, neighbor_labels[:, k:k+1], exp_dist[:, k:k+1])
        
    # Normalize weights
    init_weights = init_weights / (init_weights.sum(dim=1, keepdim=True) + 1e-8)
    
    return init_rot, init_trans, init_weights.cpu()

def get_mesh_edges(faces):
    """
    Extract unique edges from faces.
    faces: (F, 3)
    """
    if faces is None:
        return None
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges = torch.sort(edges, dim=1)[0]
    edges = torch.unique(edges, dim=0)
    return edges

def render_mesh_sequence(vertices, faces, output_path, fps=30):
    """
    Render a sequence of meshes into a video using matplotlib and ffmpeg.
    
    vertices: (T, V, 3) numpy array
    faces: (F, 3) numpy array or None
    output_path: path to save the .mp4 video
    fps: frames per second
    """
    T = vertices.shape[0]
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_render_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set up matplotlib for 3D plotting
    from mpl_toolkits.mplot3d import Axes3D
    
    # Calculate global bounds for consistent axis
    v_min = vertices.min(axis=(0, 1))
    v_max = vertices.max(axis=(0, 1))
    center = (v_min + v_max) / 2
    max_range = (v_max - v_min).max() / 2
    
    print(f"Rendering {T} frames to {temp_dir}...")
    frame_paths = []
    for t in range(T):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        v = vertices[t]
        if faces is not None:
            # If we have faces, we can render a surface (slow) or just points for now
            # For speed and simplicity in basic implementation, we use scatter
            ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, c='blue', alpha=0.5)
        else:
            ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, c='blue', alpha=0.5)
            
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.set_title(f"Frame {t}")
        
        frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
        frame_paths.append(frame_path)
        
    # Synthesize video using ffmpeg
    print(f"Synthesizing video to {output_path}...")
    input_pattern = os.path.join(temp_dir, "frame_%04d.png")
    
    ffmpeg_bin = "/usr/bin/ffmpeg"
    if not os.path.exists(ffmpeg_bin):
        ffmpeg_bin = "ffmpeg"
        
    cmd = [
        ffmpeg_bin, '-y', '-loglevel', 'error', '-r', str(fps),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        success = os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed with exit code {e.returncode}.")
        success = False
        
    if success:
        shutil.rmtree(temp_dir)
        print(f"Video saved to {output_path}")
    else:
        print(f"Failed to create video. Temp frames kept at {temp_dir}")
