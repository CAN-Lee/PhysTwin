"""
Inference script for the Hybrid Physics-Neural MPM Simulator.

Loads a best_checkpoint.pt (with ResidualPGND + physics params), runs the
full simulation, and exports particle trajectories as inference.pkl in the
same format as the original PhysTwin pipeline: ndarray of shape (T, N, 3)
in the ORIGINAL coordinate frame (no reverse_z, no auto_offset).

Usage:
    python inference_mpm.py --case_name double_lift_sloth --config configs/softbody.yaml
    python inference_mpm.py --case_name double_lift_cloth_1 --config configs/mpm_cloth.yaml
    python inference_mpm.py --case_name single_push_rope --config configs/rope.yaml
"""

import warnings
warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor", module="warp")

import os
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from phys_expert.engine.trainer_mpm import PhysExpertMPMTrainer
from phys_expert.model.bridge.anchor_bank import (
    aggregate_anchor_positions,
    build_bridge_assets_from_particles,
)


def run_inference(trainer, save_path, save_bridge_assets=False):
    """
    Run full simulation and save particle trajectories as inference.pkl.
    Coordinate transform: MPM grid -> original PhysTwin frame.
    """
    trainer.simulator.eval()

    offset = trainer.auto_offset
    reverse_z = getattr(trainer.cfg.data, 'reverse_z', False)

    init_pos = trainer.data['init_pos'].to(trainer.device) + offset
    controller_points = trainer.controller_points
    T_data = controller_points.shape[0]
    T = min(T_data, trainer.cfg.mpm.max_frames) if trainer.cfg.mpm.max_frames > 0 else T_data

    # Get optimized physical properties
    w_patch, mu_patch, lam_patch, fk_patch, fdir_patch, friction, yield_patch, E_patch, nu_patch, visc_patch = trainer.get_current_phys_props()

    def gather_and_interp(patch_data):
        flat_idx = trainer.patch_idx.squeeze(0).view(-1)
        gathered = patch_data[flat_idx].view(1, -1, 3, patch_data.shape[-1])
        return torch.sum(trainer.interp_weights * gathered, dim=2).squeeze(0)

    p_weights = gather_and_interp(w_patch)
    p_mu = gather_and_interp(mu_patch.unsqueeze(-1)).squeeze()
    p_lam = gather_and_interp(lam_patch.unsqueeze(-1)).squeeze()
    p_fk = gather_and_interp(fk_patch.unsqueeze(-1)).squeeze()
    p_fdir = torch.nn.functional.normalize(gather_and_interp(fdir_patch), dim=1, eps=1e-8)
    p_yield = gather_and_interp(yield_patch.unsqueeze(-1)).squeeze()
    p_visc = gather_and_interp(visc_patch.unsqueeze(-1)).squeeze()

    import warp as wp
    from phys_expert.model.diff_simulator.warp_solver.warp_utils import torch2warp_float, torch2warp_vec3

    trainer.simulator.reset(init_pos, controller_pos=controller_points[0])

    expert_order = ['nh', 'co', 'st', 'fi']
    active_experts_list = getattr(trainer.cfg.mpm, 'active_experts', expert_order)
    mask_active = [1 if e in active_experts_list else 0 for e in expert_order]
    active_mask_wp = wp.array(mask_active, dtype=wp.int32, device=trainer.simulator.warp_device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        moe_params_wp = {
            'weights': torch2warp_float(p_weights),
            'mu': torch2warp_float(p_mu),
            'lam': torch2warp_float(p_lam),
            'fk': torch2warp_float(p_fk),
            'fdir': torch2warp_vec3(p_fdir),
            'active_mask': active_mask_wp,
        }

    # History buffers for ResidualPGND
    H = getattr(trainer.cfg.residual if hasattr(trainer.cfg, 'residual') else None, 'n_history', 2)
    x_history = [init_pos.clone() for _ in range(H)]
    v_history = [torch.zeros_like(init_pos) for _ in range(H)]

    all_positions = []

    # Only export the original PKL particles (surface + other_surface + interior),
    # excluding Gaussian-filled particles that were added for MPM volume simulation.
    # This matches the original PhysTwin inference.pkl format and avoids OOM in LBS.
    pc = trainer.data.get('particle_counts', {})
    n_pkl = pc.get('surface', 0) + pc.get('other_surface', 0) + pc.get('interior', 0)
    n_total = init_pos.shape[0]
    if n_pkl > 0 and n_pkl < n_total:
        print(f"Exporting first {n_pkl} particles (excluding {n_total - n_pkl} Gaussian-filled)")
    else:
        n_pkl = n_total
        print(f"Exporting all {n_pkl} particles")

    def to_original_frame(pos_shifted):
        """Convert from auto_offset+reverse_z frame to original PhysTwin frame."""
        pos = (pos_shifted[:n_pkl].detach() - offset).cpu()
        if reverse_z:
            pos[..., 2] *= -1.0
        return pos

    def to_original_frame_full(pos_shifted):
        pos = (pos_shifted.detach() - offset).cpu()
        if reverse_z:
            pos[..., 2] *= -1.0
        return pos

    # Frame 0: save initial state before any simulation (matches original PhysTwin format)
    init_x = (trainer.simulator.x - trainer.simulator.shift)
    all_positions.append(to_original_frame(init_x))
    bridge_anchor_traj = []
    bridge_assets = None
    bridge_cfg = getattr(trainer.cfg, "bridge", None)
    if save_bridge_assets:
        init_full_world = to_original_frame_full(init_x)
        bridge_assets = build_bridge_assets_from_particles(
            particles_t0=init_full_world,
            scene_id=trainer.scene_id,
            reverse_z=reverse_z,
            num_render_anchors=int(getattr(bridge_cfg, "num_render_anchors", 4096)),
            anchor_particle_k=int(getattr(bridge_cfg, "anchor_particle_k", 16)),
            anchor_graph_k=int(getattr(bridge_cfg, "anchor_graph_k", 8)),
            anchor_particle_temp=float(getattr(bridge_cfg, "anchor_particle_temp", 0.01)),
            anchor_traj=None,
        )
        bridge_anchor_traj.append(
            aggregate_anchor_positions(
                init_full_world,
                bridge_assets.anchor_particle_indices,
                bridge_assets.anchor_particle_weights,
            )
        )

    with torch.no_grad():
        for t in tqdm(range(1, T), desc="MPM Inference"):
            c_pos_end = controller_points[t]
            c_pos_start = controller_points[t - 1]
            v_ctrl_t = (c_pos_end - c_pos_start) / (trainer.cfg.mpm.dt * trainer.cfg.mpm.steps_per_frame)

            x_start_frame = (trainer.simulator.x - trainer.simulator.shift).detach().unsqueeze(0)

            K_damping = getattr(trainer.cfg.residual, 'damping_interval', 20)
            current_damping = None
            res_mode = getattr(trainer.cfg.residual, 'mode', 'both')

            for s in range(trainer.cfg.mpm.steps_per_frame):
                alpha = (s + 1) / trainer.cfg.mpm.steps_per_frame
                curr_target_pos = c_pos_start + alpha * (c_pos_end - c_pos_start)

                if trainer.residual_net is not None and s % K_damping == 0 and res_mode in ['damping', 'both']:
                    x_his_tensor = torch.stack(x_history, dim=1).unsqueeze(0)
                    v_his_tensor = torch.stack(v_history, dim=1).unsqueeze(0)
                    curr_x_mpm = (trainer.simulator.x - trainer.simulator.shift).unsqueeze(0)
                    curr_v_mpm = trainer.simulator.v.unsqueeze(0)
                    current_damping = trainer.residual_net(
                        curr_x_mpm, curr_v_mpm, x_start_frame,
                        x_his_tensor, v_his_tensor, mode="damping"
                    ).squeeze(0)

                x_curr = trainer.simulator.step(
                    moe_params_wp,
                    controller_pos=curr_target_pos,
                    controller_vel=v_ctrl_t,
                    residual_v=None,
                    damping_override=current_damping,
                )

            # Neural feedback correction
            if trainer.residual_net is not None and res_mode in ['residual', 'both']:
                trainer.residual_net.eval()
                x_his_tensor = torch.stack(x_history, dim=1).unsqueeze(0)
                v_his_tensor = torch.stack(v_history, dim=1).unsqueeze(0)
                curr_x_mpm = (trainer.simulator.x - trainer.simulator.shift).unsqueeze(0)
                curr_v_mpm = trainer.simulator.v.unsqueeze(0)
                delta_v = trainer.residual_net(
                    curr_x_mpm, curr_v_mpm, x_start_frame,
                    x_his_tensor, v_his_tensor, mode="residual"
                ).squeeze(0)
                trainer.simulator.v = trainer.simulator.v + delta_v
                frame_dt = trainer.cfg.mpm.dt * trainer.cfg.mpm.steps_per_frame
                trainer.simulator.x = trainer.simulator.x + delta_v * frame_dt
                x_curr = trainer.simulator.x - trainer.simulator.shift

                x_history.pop(0)
                x_history.append((trainer.simulator.x - trainer.simulator.shift).detach())
                v_history.pop(0)
                v_history.append(trainer.simulator.v.detach())

            trainer.simulator.x = trainer.simulator.x.detach()
            trainer.simulator.v = trainer.simulator.v.detach()
            trainer.simulator.F = trainer.simulator.F.detach()
            trainer.simulator.C = trainer.simulator.C.detach()

            all_positions.append(to_original_frame(x_curr))
            if bridge_assets is not None:
                x_curr_world = to_original_frame_full(x_curr)
                bridge_anchor_traj.append(
                    aggregate_anchor_positions(
                        x_curr_world,
                        bridge_assets.anchor_particle_indices,
                        bridge_assets.anchor_particle_weights,
                    )
                )

    trajectory = torch.stack(all_positions, dim=0).numpy().astype(np.float32)  # (T, N, 3)
    print(f"Trajectory shape: {trajectory.shape}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajectory, f)
    print(f"Saved inference.pkl to {save_path}")

    if bridge_assets is not None:
        bridge_assets.anchor_traj = torch.stack(bridge_anchor_traj, dim=0)
        bridge_path = os.path.join(os.path.dirname(save_path), "bridge_assets.pt")
        torch.save(bridge_assets.to_dict(), bridge_path)
        print(f"Saved bridge_assets.pt to {bridge_path}")


def main():
    parser = argparse.ArgumentParser(description="MPM Hybrid Simulator Inference -> inference.pkl")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="Config YAML used during training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Default: {output_dir}/{case_name}/best_checkpoint.pt")
    parser.add_argument("--output_dir", type=str, default="./output_3/mpm_inference",
                        help="Directory to save inference.pkl (flat: {output_dir}/{case_name}/inference.pkl)")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--save_bridge_assets", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = OmegaConf.load(args.config)
    cfg.mpm.device = 'cuda'

    # Resolve checkpoint path
    if args.checkpoint is None:
        ckpt_path = os.path.join(cfg.output_dir, args.case_name, "best_checkpoint.pt")
    else:
        ckpt_path = args.checkpoint

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    print(f"=== MPM Inference: {args.case_name} ===")
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {ckpt_path}")

    trainer = PhysExpertMPMTrainer(cfg, args.case_name)
    trainer.load_from_checkpoint(ckpt_path)

    save_path = os.path.join(args.output_dir, args.case_name, "inference.pkl")
    run_inference(trainer, save_path, save_bridge_assets=args.save_bridge_assets)


if __name__ == "__main__":
    main()
