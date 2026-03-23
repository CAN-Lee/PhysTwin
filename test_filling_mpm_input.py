"""
Test Gaussian filling on the same input point cloud used by train_mpm.py / trainer_mpm.py.

Loads the case via the same data path and init_pos construction as PhysTwinDataset,
then runs PhysFlow-style filling from gaussian_output when available, and reports
particle counts (MPM default vs Gaussian-filled) and logs to TensorBoard for comparison.
"""

import os
import argparse
import pickle
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


def load_mpm_init_pos(data_root, case_name, reverse_z=False, max_frames=0):
    """
    Same logic as phys_expert/data/dataset_mpm.py: load final_data.pkl and build init_pos.
    init_pos = object_points[0] + surface_points + interior_points.
    """
    scene_path = os.path.join(data_root, case_name)
    data_path = os.path.join(scene_path, "final_data.pkl")
    if not os.path.isfile(data_path):
        return None, None

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    object_points = np.asarray(data["object_points"], dtype=np.float32)
    if max_frames > 0:
        object_points = object_points[:max_frames]
    interior_points = np.asarray(data["interior_points"], dtype=np.float32)
    other_surface = np.asarray(data["surface_points"], dtype=np.float32) if "surface_points" in data else np.empty((0, 3))

    if reverse_z:
        object_points[..., 2] *= -1.0
        interior_points[..., 2] *= -1.0
        other_surface[..., 2] *= -1.0

    init_pos = np.concatenate([object_points[0], other_surface, interior_points], axis=0)
    n_surf = object_points[0].shape[0] + other_surface.shape[0]
    n_interior_pkl = interior_points.shape[0]
    return init_pos, {"n_surf": n_surf, "n_interior_pkl": n_interior_pkl, "object_points": object_points}


def log_mesh_to_tensorboard(orig_pos, filled_pos, log_dir, tag_prefix="", n_filled_shell=None,
                            init_pos_mpm=None, n_new_internal=0,
                            init_pos_pkl_dense=None, n_dense=0):
    """
    Log point clouds to TensorBoard:
    1) MPM_init_pos: pkl only (blue).
    2) Gaussian_filled: full GS fill — shell blue, all new (dense+internal) red; total can be 10w+.
    3) MPM_init_pos_pkl_dense: pkl + dense only (blue + last n_dense red).
    4) MPM_init_pos_used: what MPM actually uses = pkl + new_internal only (blue + last n_new_internal red).
    """
    writer = SummaryWriter(log_dir=log_dir)
    v_orig = torch.from_numpy(np.asarray(orig_pos).astype(np.float32)).unsqueeze(0)
    if hasattr(filled_pos, "numpy"):
        filled_pos = filled_pos.numpy()
    v_filled = np.asarray(filled_pos).astype(np.float32)
    n_shell_f = (n_filled_shell if n_filled_shell is not None else v_filled.shape[0])

    # 1) MPM init from pkl only (blue)
    c_o = torch.zeros(1, v_orig.shape[1], 3, dtype=torch.uint8)
    c_o[..., 2] = 255
    writer.add_mesh(f"{tag_prefix}MPM_init_pos", vertices=v_orig, colors=c_o, global_step=0)

    # 2) Full Gaussian filled: shell blue, new (dense+internal) red — so red count = n_new (e.g. 4078), not 84
    v_f = torch.from_numpy(v_filled).float().unsqueeze(0)
    c_f = torch.zeros(1, v_f.shape[1], 3, dtype=torch.uint8)
    c_f[0, :n_shell_f, 2] = 255
    c_f[0, n_shell_f:, 0] = 255
    writer.add_mesh(f"{tag_prefix}Gaussian_filled", vertices=v_f, colors=c_f, global_step=0)

    # 3) pkl + dense only (last n_dense points in red)
    if init_pos_pkl_dense is not None and init_pos_pkl_dense.shape[0] > 0 and n_dense > 0:
        v_pd = torch.from_numpy(np.asarray(init_pos_pkl_dense).astype(np.float32)).unsqueeze(0)
        c_pd = torch.zeros(1, v_pd.shape[1], 3, dtype=torch.uint8)
        c_pd[..., 2] = 255
        c_pd[0, -n_dense:, 0] = 255
        c_pd[0, -n_dense:, 2] = 0
        writer.add_mesh(f"{tag_prefix}MPM_init_pos_pkl_dense", vertices=v_pd, colors=c_pd, global_step=0)

    # 4) What MPM actually uses: pkl + new_internal only (last n_new_internal points in red)
    if init_pos_mpm is not None and init_pos_mpm.shape[0] > 0 and n_new_internal > 0:
        v_mpm = torch.from_numpy(np.asarray(init_pos_mpm).astype(np.float32)).unsqueeze(0)
        c_mpm = torch.zeros(1, v_mpm.shape[1], 3, dtype=torch.uint8)
        c_mpm[..., 2] = 255
        c_mpm[0, -n_new_internal:, 0] = 255
        c_mpm[0, -n_new_internal:, 2] = 0
        writer.add_mesh(f"{tag_prefix}MPM_init_pos_used", vertices=v_mpm, colors=c_mpm, global_step=0)

    writer.close()
    print(f"TensorBoard logged to {log_dir}")
    print(f"  Tags: MPM_init_pos (pkl), Gaussian_filled (shell+dense+internal), MPM_init_pos_pkl_dense (pkl+dense, red={n_dense}), MPM_init_pos_used (pkl+internal, red={n_new_internal})")


def main():
    parser = argparse.ArgumentParser(description="Test Gaussian filling on MPM trainer input")
    parser.add_argument("--case_name", type=str, required=True, help="Same as train_mpm.py --case_name")
    parser.add_argument("--config", type=str, default="configs/softbody.yaml", help="Config for data.root and data.reverse_z")
    parser.add_argument("--gaussian_output_dir", type=str, default="gaussian_output")
    parser.add_argument("--no_tensorboard", action="store_true", help="Skip TensorBoard logging")
    parser.add_argument("--grid_n", type=int, default=80)
    parser.add_argument("--density_thres", type=float, default=0.8)
    parser.add_argument("--search_thres", type=float, default=0.5)
    parser.add_argument("--max_particles_per_cell", type=int, default=2)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_root = cfg.get("data", {}).get("root", "data/different_types")
    reverse_z = cfg.get("data", {}).get("reverse_z", False)
    max_frames = int(cfg.get("mpm", {}).get("max_frames", 0))

    # 1. Load MPM default init_pos (same as trainer_mpm / dataset_mpm)
    init_pos, meta = load_mpm_init_pos(data_root, args.case_name, reverse_z=reverse_z, max_frames=max_frames)
    if init_pos is None:
        print(f"Error: No final_data.pkl at {data_root}/{args.case_name}")
        return

    n_mpm = len(init_pos)
    print(f"[MPM default] init_pos count: {n_mpm} (surface+other: {meta['n_surf']}, interior from pkl: {meta['n_interior_pkl']})")

    # 2. Gaussian filling when PLY exists
    filled_pos_np = None
    n_new = 0
    used_gaussian = False
    try:
        from gaussian_filling import get_gaussian_ply_path, fill_particles_gaussian
        ply_path = get_gaussian_ply_path(args.gaussian_output_dir, args.case_name)
    except ImportError:
        ply_path = None

    new_internal_pos = None
    if ply_path is not None:
        print(f"Using Gaussian filling from {ply_path}")
        filled_pos, n_new, new_internal_pos, _ = fill_particles_gaussian(
            ply_path,
            grid_n=args.grid_n,
            density_thres=args.density_thres,
            search_thres=args.search_thres,
            max_particles_per_cell=args.max_particles_per_cell,
            padding=0.1,
        )
        filled_pos_np = filled_pos.numpy()
        n_shell_gs = filled_pos_np.shape[0] - n_new
        n_internal_only = new_internal_pos.shape[0]
        # 与 pkl 一致：若 config 里 reverse_z，则对 Gaussian 结果也做 z 翻转，避免 pkl 与 dense/internal 分层
        if reverse_z:
            filled_pos_np = filled_pos_np.copy()
            filled_pos_np[:, 2] *= -1.0
            new_internal_pos = new_internal_pos.copy()
            new_internal_pos[:, 2] *= -1.0
        used_gaussian = True
        print(f"[Gaussian filled] shell (from GS): {n_shell_gs}, new dense+internal: {n_new}, new internal only: {n_internal_only}, full total: {len(filled_pos_np)}")
    else:
        print(f"No gaussian_output PLY for {args.case_name}; skipping Gaussian filling.")

    # 3. MPM init_pos = pkl (surface+interior) + Gaussian new internal only (no full shell)
    if used_gaussian and new_internal_pos is not None and new_internal_pos.shape[0] > 0:
        init_pos_mpm = np.concatenate([init_pos, new_internal_pos], axis=0)
        print(f"\n[MPM init_pos] pkl only: {n_mpm} -> with Gaussian new internal: {len(init_pos_mpm)} (added {new_internal_pos.shape[0]})")
    else:
        init_pos_mpm = init_pos
        if used_gaussian:
            print(f"\n[MPM init_pos] using pkl only: {n_mpm} (no new internal particles added)")

    # 4. pkl + dense only (for TensorBoard)，dense 需与 pkl 同一坐标系（reverse_z）
    n_dense = 0
    init_pos_pkl_dense = None
    if used_gaussian and filled_pos_np is not None and new_internal_pos is not None:
        n_internal_only = new_internal_pos.shape[0]
        n_dense = n_new - n_internal_only
        if n_dense > 0:
            new_dense_pos = filled_pos_np[n_shell_gs : n_shell_gs + n_dense]
            init_pos_pkl_dense = np.concatenate([init_pos, new_dense_pos], axis=0)

    # 5. Summary and TensorBoard: log pkl init, full Gaussian filled, pkl+dense, and MPM init actually used
    if used_gaussian and filled_pos_np is not None:
        if not args.no_tensorboard:
            log_dir = os.path.join("runs", "particle_filling_mpm_input", args.case_name)
            os.makedirs(log_dir, exist_ok=True)
            n_shell_gs = len(filled_pos_np) - n_new
            n_added = new_internal_pos.shape[0] if new_internal_pos is not None else 0
            log_mesh_to_tensorboard(
                init_pos, filled_pos_np, log_dir, tag_prefix=f"{args.case_name}_",
                n_filled_shell=n_shell_gs,
                init_pos_mpm=init_pos_mpm if used_gaussian else None,
                n_new_internal=n_added,
                init_pos_pkl_dense=init_pos_pkl_dense,
                n_dense=n_dense,
            )


if __name__ == "__main__":
    main()
