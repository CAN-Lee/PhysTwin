import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label as ndi_label
from torch.utils.tensorboard import SummaryWriter
import os

class VoxelFiller:
    def __init__(self, grid_res=64, padding=0.1, jitter=0.8, dilation_iterations=2, max_interior_ratio=1.0):
        self.grid_res = grid_res
        self.padding = padding
        self.jitter = jitter
        self.dilation_iterations = max(1, int(dilation_iterations))
        # 内部点数量不超过表面点的 max_interior_ratio 倍，避免填充点完全盖住表面
        self.max_interior_ratio = max(0.1, float(max_interior_ratio))

    def fill(self, pos):
        if isinstance(pos, torch.Tensor):
            pos_np = pos.detach().cpu().numpy()
        else:
            pos_np = pos

        p_min = pos_np.min(axis=0)
        p_max = pos_np.max(axis=0)
        p_range = p_max - p_min
        p_min -= p_range * self.padding
        p_max += p_range * self.padding
        p_range = p_max - p_min
        dx = p_range.max() / (self.grid_res - 1)

        indices = ((pos_np - p_min) / dx).astype(int)
        indices = np.clip(indices, 0, self.grid_res - 1)
        grid = np.zeros((self.grid_res, self.grid_res, self.grid_res), dtype=bool)
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # 1. Dilation to close small gaps in the voxelized shell
        dilated_grid = binary_dilation(grid, iterations=self.dilation_iterations)
        background = ~dilated_grid
        # 2. Interior = background voxels connected to centroid (flood-fill from inside)
        #    Centroid of surface is almost always inside a closed shape.
        centroid = pos_np.mean(axis=0)
        seed_ijk = np.clip(
            ((centroid - p_min) / dx).astype(int), 0, self.grid_res - 1
        )
        ci, cj, ck = seed_ijk[0], seed_ijk[1], seed_ijk[2]
        if dilated_grid[ci, cj, ck]:
            # Centroid voxel is on the shell; pick a neighbor toward center (e.g. try all 6)
            for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                ni, nj, nk = ci + di, cj + dj, ck + dk
                if 0 <= ni < self.grid_res and 0 <= nj < self.grid_res and 0 <= nk < self.grid_res and background[ni, nj, nk]:
                    ci, cj, ck = ni, nj, nk
                    break
            else:
                return torch.from_numpy(pos_np).float(), 0
        # 3. Label background by 6-connectivity
        struct = np.zeros((3, 3, 3), dtype=bool)
        struct[1, 1, :] = struct[1, :, 1] = struct[:, 1, 1] = True
        labels, ncomp = ndi_label(background, structure=struct)
        seed_label = labels[ci, cj, ck]
        if seed_label == 0:
            return torch.from_numpy(pos_np).float(), 0
        # 4. 只接受“不与网格边界连通”的组分作为内部（否则壳未闭合，质心在“外部”）
        exterior_labels = set()
        for i in [0, self.grid_res - 1]:
            exterior_labels.update(labels[i, :, :][background[i, :, :]].flat)
        for j in [0, self.grid_res - 1]:
            exterior_labels.update(labels[:, j, :][background[:, j, :]].flat)
        for k in [0, self.grid_res - 1]:
            exterior_labels.update(labels[:, :, k][background[:, :, k]].flat)
        exterior_labels.discard(0)
        if seed_label in exterior_labels:
            # 质心所在组分与网格边界连通 → 体素壳未闭合，得到的是外部，拒绝
            import sys
            print("VoxelFiller: shell not closed (interior connected to grid border). Try larger dilation_iterations or grid_res.", file=sys.stderr)
            return torch.from_numpy(pos_np).float(), 0
        internal_mask = (labels == seed_label)
        internal_indices = np.argwhere(internal_mask)

        if len(internal_indices) == 0:
            return torch.from_numpy(pos_np).float(), 0

        # 下采样内部点，避免数量远多于表面点导致完全盖住表面
        n_surface = len(pos_np)
        max_interior = max(1, int(n_surface * self.max_interior_ratio))
        if len(internal_indices) > max_interior:
            rng = np.random.default_rng()
            idx_choice = rng.choice(len(internal_indices), size=max_interior, replace=False)
            internal_indices = internal_indices[idx_choice]

        jitter_vals = (np.random.rand(*internal_indices.shape) - 0.5) * self.jitter
        new_pos_np = (internal_indices + 0.5 + jitter_vals) * dx + p_min
        
        filled_pos = np.concatenate([pos_np, new_pos_np], axis=0)
        return torch.from_numpy(filled_pos).float(), len(new_pos_np)

def visualize_filling(orig_pos, filled_pos, output_path):
    fig = plt.figure(figsize=(12, 6))
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(orig_pos[:, 0], orig_pos[:, 1], orig_pos[:, 2], s=2, c='blue', alpha=0.5)
    ax1.set_title(f"Original Shell (N={len(orig_pos)})")
    
    # Filled: 先画内部点（底层），再画表面点（上层更显眼）
    ax2 = fig.add_subplot(122, projection='3d')
    new_count = len(filled_pos) - len(orig_pos)
    if new_count > 0:
        ax2.scatter(filled_pos[len(orig_pos):, 0], filled_pos[len(orig_pos):, 1], filled_pos[len(orig_pos):, 2], s=1, c='red', alpha=0.4, label='Internal')
    ax2.scatter(filled_pos[:len(orig_pos), 0], filled_pos[:len(orig_pos), 1], filled_pos[:len(orig_pos), 2], s=4, c='blue', alpha=0.7, label='Shell')
    ax2.set_title(f"Filled Softbody (Shell={len(orig_pos)}, Internal={new_count})")
    ax2.legend()
    
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


def log_mesh_to_tensorboard(orig_pos, filled_pos, log_dir="runs/particle_filling"):
    """Log point clouds as mesh to TensorBoard for 3D visualization."""
    writer = SummaryWriter(log_dir=log_dir)

    # Original shell: vertices [1, N, 3], colors blue [0,0,255]
    v_orig = torch.from_numpy(orig_pos).float().unsqueeze(0)  # [1, N, 3]
    c_orig = torch.zeros(1, len(orig_pos), 3, dtype=torch.uint8)
    c_orig[..., 2] = 255  # BGR in TensorBoard -> blue
    writer.add_mesh("Original_Shell", vertices=v_orig, colors=c_orig, global_step=0)

    # Filled: shell (blue) + internal (red)
    v_filled = torch.from_numpy(filled_pos).float().unsqueeze(0)  # [1, M, 3]
    c_filled = torch.zeros(1, len(filled_pos), 3, dtype=torch.uint8)
    n_orig = len(orig_pos)
    c_filled[0, :n_orig, 2] = 255   # shell -> blue
    c_filled[0, n_orig:, 0] = 255   # internal -> red
    writer.add_mesh("Filled_Softbody", vertices=v_filled, colors=c_filled, global_step=0)

    writer.close()
    print(f"TensorBoard mesh logged to {log_dir}. Run: tensorboard --logdir {log_dir}")

def fill_with_gaussian_if_available(case_name, gaussian_output_dir="gaussian_output", exp_name="default", **kwargs):
    """
    If gaussian_output has a PLY for this case, run PhysFlow-style filling and return (filled_pos, n_new).
    Otherwise return None so caller can fall back to VoxelFiller.
    """
    try:
        from gaussian_filling import get_gaussian_ply_path, fill_particles_gaussian
    except ImportError:
        return None
    ply_path = get_gaussian_ply_path(gaussian_output_dir, case_name, exp_name=exp_name)
    if ply_path is None:
        return None
    filled_pos, n_new, _, _ = fill_particles_gaussian(ply_path, **kwargs)
    return filled_pos.numpy(), n_new


if __name__ == "__main__":
    test_pkl = "data/different_types/double_lift_sloth/final_data.pkl"
    gaussian_output_dir = "gaussian_output"
    case_name = "double_lift_sloth"

    # Prefer PhysFlow-style filling from gaussian_output when available
    filled_pos_np = None
    orig_pos = None
    new_count = 0
    used_gaussian = False

    ply_path = None
    try:
        from gaussian_filling import get_gaussian_ply_path, fill_particles_gaussian
        ply_path = get_gaussian_ply_path(gaussian_output_dir, case_name)
    except ImportError:
        pass

    if ply_path is not None:
        print(f"Using Gaussian filling from {ply_path}")
        # 填充密度由以下阈值控制（在 gaussian_filling.fill_particles_gaussian 中）:
        # - density_thres: 壳密度阈值，越小壳越厚、填充越多（建议 0.5~2.0）
        # - search_thres:  内部判定壳阈值，越小内部格子越多（建议 0.3~1.0）
        # - max_particles_per_cell: 每格粒子数，越大总填充越多（1~4）
        # - grid_n: 网格分辨率，越大越密但越慢（64~128）
        filled_pos, new_count, _, _ = fill_particles_gaussian(
            ply_path,
            grid_n=80,
            density_thres=0.8,
            search_thres=0.5,
            max_particles_per_cell=2,
            padding=0.1,
        )
        filled_pos_np = filled_pos.numpy()
        orig_pos = filled_pos_np[:filled_pos_np.shape[0] - new_count]
        used_gaussian = True
    elif os.path.exists(test_pkl):
        with open(test_pkl, 'rb') as f:
            data = pickle.load(f)
        if 'object_points' in data:
            orig_pos = data['object_points'][0]
        else:
            surf = data.get('surface_points', np.zeros((0, 3)))
            interior = data.get('interior_points', np.zeros((0, 3)))
            orig_pos = np.concatenate([surf, interior], axis=0)
        filler = VoxelFiller(grid_res=48, dilation_iterations=3)
        filled_pos, new_count = filler.fill(orig_pos)
        filled_pos_np = filled_pos.numpy()
    else:
        print(f"Error: {test_pkl} not found and no gaussian_output for {case_name}.")

    if filled_pos_np is not None and orig_pos is not None:
        print(f"Original/shell particles: {len(orig_pos)}")
        print(f"New internal particles: {new_count}")
        if used_gaussian:
            print("(PhysFlow-style filling from gaussian_output)")
        log_mesh_to_tensorboard(orig_pos, filled_pos_np, log_dir="runs/particle_filling")
