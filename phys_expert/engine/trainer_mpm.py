import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import pickle
import shutil
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.loss import chamfer_distance

# Use Agg backend for headless rendering
import matplotlib
matplotlib.use('Agg')

from ..engine.simulator_mpm import MPMSimulator
from ..data.dataset_mpm import PhysTwinDataset
from ..utils.mpm_utils import youngs_poisson_to_lame
from ..model.residual_pgnd import ResidualPGND

class PhysExpertMPMTrainer:
    """
    Stage 1: System Identification.
    Directly optimize physical parameters for a specific scene using Differentiable MPM.
    """
    def __init__(self, cfg, scene_id, resume_path=None):
        self.cfg = cfg
        self.device = torch.device(cfg.mpm.device)
        self.scene_id = scene_id
        
        # [NEW] Save config for reproducibility (At Start)
        os.makedirs(os.path.join(cfg.output_dir, scene_id), exist_ok=True)
        config_save_path = os.path.join(cfg.output_dir, scene_id, "config.yaml")
        OmegaConf.save(cfg, config_save_path)
        print(f"Config saved to {config_save_path}")
        
        # 1. Setup Data
        self.dataset = PhysTwinDataset(cfg, case_name=scene_id)
        if len(self.dataset) == 0:
            raise ValueError(f"Scene {scene_id} not found!")
        self.data = self.dataset[0]
        
        # 2. Initialize Learnable Parameters (at Patch Level)
        # [FIXED] Use cfg.router.static.n_patches if available, else default
        if hasattr(cfg, 'router') and hasattr(cfg.router, 'static'):
             self.n_patches = cfg.router.static.n_patches
        else:
             # Fallback if 'router' config is removed
             self.n_patches = 64
        self.active_experts = getattr(cfg.mpm, 'active_experts', ['nh', 'co', 'st', 'fi'])
        num_experts = len(self.active_experts)
        
        # [NEW] Track best loss for saving best checkpoint
        self.best_loss = float('inf')
        
        # Expert Weights: [K, num_active]
        # [NEW] Load initial weights from config if available, else uniform
        config_weights = getattr(cfg.mpm, 'init_weights', None)
        if config_weights and len(config_weights) == num_experts:
            # Normalize to sum to 1 just in case
            w_tensor = torch.tensor(config_weights, device=self.device)
            w_tensor = w_tensor / w_tensor.sum()
            # Convert to log space because we optimize log_weights
            # Add epsilon to avoid log(0)
            init_log_weights = torch.log(w_tensor + 1e-6)
            # Expand to all patches: [1, num_experts] -> [n_patches, num_experts]
            self.log_weights = nn.Parameter(init_log_weights.unsqueeze(0).repeat(self.n_patches, 1))
            print(f"Initialized expert weights from config: {config_weights}")
        else:
            init_weights = torch.ones(self.n_patches, num_experts) / num_experts
            self.log_weights = nn.Parameter(torch.log(init_weights).to(self.device))
            print("Initialized expert weights uniformly.")
        
        # [REVISED] 精细化初始化：优先使用特定参数的初始值，否则退回到 init_raw_params
        init_val = getattr(cfg.mpm, 'init_raw_params', 0.0)
        
        def get_init(name, default):
            return getattr(cfg.mpm, f'init_raw_{name}', default)

        self.raw_E = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('E', init_val))
        self.raw_nu = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('nu', init_val))
        self.raw_fiber_k = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('fiber_k', init_val))
        self.raw_yield = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('yield', init_val))
        self.raw_viscosity = nn.Parameter(torch.ones(self.n_patches, 1, device=self.device) * get_init('viscosity', init_val))
        
        self.raw_fiber_dir = nn.Parameter(torch.randn(self.n_patches, 3, device=self.device) * 0.1)

        # 3. [NEW] Resume Path Search (Moved earlier to decide log_dir)
        self.resume_checkpoint = None
        self.resume_path = resume_path
        
        # Priority: explicit resume_path > checkpoint_*.pt > optimized_params.pkl
        if self.resume_path is None:
            # 1. Check for full checkpoints (*.pt)
            log_base = os.path.join(cfg.output_dir, scene_id, 'mpm_train')
            pt_checkpoints = []
            if os.path.exists(log_base):
                for root, dirs, files in os.walk(log_base):
                    for f in files:
                        if f.startswith("checkpoint_iter_") and f.endswith(".pt"):
                            full_path = os.path.join(root, f)
                            try:
                                iter_num = int(f.split('_')[-1].split('.')[0])
                                pt_checkpoints.append((iter_num, full_path))
                            except ValueError:
                                continue
            
            # 2. Check for final checkpoint
            final_ckpt = os.path.join(cfg.output_dir, scene_id, "final_checkpoint.pt")
            if os.path.exists(final_ckpt):
                 self.resume_path = final_ckpt
            elif pt_checkpoints:
                 self.resume_path = sorted(pt_checkpoints, key=lambda x: x[0])[-1][1]
            else:
                # Fallback to old pkl logic
                opt_path = os.path.join(cfg.output_dir, scene_id, "optimized_params.pkl")
                if os.path.exists(opt_path):
                    self.resume_path = opt_path
                else:
                    # Look for params_iter_*.pkl
                    if os.path.exists(log_base):
                        checkpoints = []
                        for root, dirs, files in os.walk(log_base):
                            for f in files:
                                if f.startswith("params_iter_") and f.endswith(".pkl"):
                                    full_path = os.path.join(root, f)
                                    try:
                                        iter_num = int(f.split('_')[-1].split('.')[0])
                                        checkpoints.append((iter_num, full_path))
                                    except ValueError:
                                        continue
                        if checkpoints:
                            self.resume_path = sorted(checkpoints, key=lambda x: x[0])[-1][1]

        # 4. [NEW] Logging Initialization (Scheme A: Reuse existing log_dir if resuming)
        self.log_dir = None
        if self.resume_path and os.path.exists(self.resume_path):
            # Try to determine existing log_dir from resume_path
            # Standard path: .../mpm_train/TIMESTAMP/checkpoint_iter_N.pt
            parent_dir = os.path.dirname(self.resume_path)
            if 'mpm_train' in parent_dir:
                # If resume_path is inside a timestamp folder (not log_base itself)
                log_base = os.path.join(cfg.output_dir, scene_id, 'mpm_train')
                if parent_dir != log_base and os.path.dirname(parent_dir) == log_base:
                    self.log_dir = parent_dir
                    print(f"[RESUME] Reusing existing log directory: {self.log_dir}")
            
            # If still not found (e.g. final_checkpoint.pt in scene root), 
            # try to find the latest folder in mpm_train
            if self.log_dir is None:
                log_base = os.path.join(cfg.output_dir, scene_id, 'mpm_train')
                if os.path.exists(log_base):
                    subdirs = sorted([os.path.join(log_base, d) for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d))])
                    if subdirs:
                        self.log_dir = subdirs[-1]
                        print(f"[RESUME] Found latest log directory: {self.log_dir}")

        if self.log_dir is None:
            # Fresh start or could not find existing log_dir
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_dir = os.path.join(cfg.output_dir, scene_id, 'mpm_train', timestamp)
            print(f"[INFO] Created new log directory: {self.log_dir}")

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.simulator = MPMSimulator(cfg.mpm).to(self.device)
        self.simulator.debug_mode = True # [DEBUG] Enable velocity clamping inspection
        
        # 4. [NEW] Initialize Residual PGND
        if hasattr(cfg, 'residual'):
            self.residual_net = ResidualPGND(cfg.residual).to(self.device)
            print("[INFO] ResidualPGND initialized.")
        else:
            self.residual_net = None
            print("[INFO] No residual config found, ResidualPGND disabled.")

        # Actually load the checkpoint weights
        if self.resume_path and os.path.exists(self.resume_path):
            self.load_from_checkpoint(self.resume_path)

        # [STABILITY] Gradient Hook: Automatically zero out NaN/Inf gradients and clamp large ones
        def nan_to_zero(grad):
            grad = torch.where(torch.isnan(grad) | torch.isinf(grad), torch.zeros_like(grad), grad)
            return torch.clamp(grad, -1e4, 1e4)
            
        protected_params = [self.log_weights, self.raw_E, self.raw_nu, self.raw_fiber_k, self.raw_fiber_dir, self.raw_yield, self.raw_viscosity]
        if self.residual_net is not None:
            protected_params.extend(list(self.residual_net.parameters()))
            
        for p in protected_params:
            p.register_hook(nan_to_zero)
        
        # 5. [NEW] Automatic Centering Logic
        # Calculate bounding box across all frames to find the best centering offset
        obj_pts = self.data['init_pos'].to(self.device) # [N, 3]
        gt_pts = self.data['gt_surface_tracks'].to(self.device).view(-1, 3) # [T*N, 3]
        ctrl_pts = self.data['controller_points'].to(self.device).view(-1, 3) # [T*C, 3]
        
        all_pts = torch.cat([obj_pts, gt_pts, ctrl_pts], dim=0)
        p_min = all_pts.min(dim=0)[0]
        p_max = all_pts.max(dim=0)[0]
        
        # [REVISED] Improved Auto-centering Strategy
        # X and Y: Center the entire sequence to avoid side-wall collisions
        # Z: Set offset so the global minimum Z is at a safe height (0.15) in simulation space
        self.auto_offset = torch.zeros(3, device=self.device)
        self.auto_offset[0] = -(p_min[0] + p_max[0]) / 2.0
        self.auto_offset[1] = -(p_min[1] + p_max[1]) / 2.0
        # Target Z_sim = 0.15. Since Simulator adds 0.5 internally: 
        # Z_orig + auto_offset.z + 0.5 = 0.15 => auto_offset.z = 0.15 - 0.5 - p_min[2]
        self.auto_offset[2] = 0.15 - 0.5 - p_min[2]
        
        # Sync offset to simulator and refresh ground boundary
        self.simulator.base_offset = self.auto_offset
        self.simulator._apply_boundary()
        
        # Check if the scaled object fits in [0.05, 0.95]
        max_span = (p_max - p_min).max().item()
        
        # Log setup info to TensorBoard instead of printing to terminal
        setup_text = f"**Scene ID**: {scene_id}  \n"
        setup_text += f"**Auto-centering Offset**: {self.auto_offset.tolist()}  \n"
        setup_text += f"**Original BBox Span**: {max_span:.4f}  \n"
        
        # Establishment of controller connections is done during simulator.reset()
        # We need a quick reset to check initial connections
        self.simulator.reset(obj_pts, controller_pos=self.data['controller_points'][0].to(self.device))
        setup_text += f"**Controller Connections**: {self.simulator.num_connections}  \n"
        
        self.writer.add_text('Setup/Info', setup_text, 0)
        
        if max_span > 0.85:
            self.writer.add_text('Setup/Warnings', f"WARNING: Scene is very large (span: {max_span:.2f}).", 0)

        # [NEW] Smooth Controller Trajectory to filter out tracking noise
        # Using a moving average window from config
        self.controller_points = (self.data['controller_points'].to(self.device) + self.auto_offset)
        T_ctrl = self.controller_points.shape[0]
        window = getattr(self.cfg.mpm, 'controller_smooth_window', 5) # Use YAML value
        
        if window > 1:
            smoothed = self.controller_points.clone()
            for t in range(T_ctrl):
                t_start = max(0, t - window // 2)
                t_end = min(T_ctrl, t + window // 2 + 1)
                smoothed[t] = self.controller_points[t_start:t_end].mean(dim=0)
            self.controller_points = smoothed
            print(f"[INFO] Controller trajectory smoothed (window={window})")

        # 6. [NEW] Deterministic Patch Assignment
        # We assign each particle to a patch once at the start. 
        # This MUST be persistent to ensure learned parameters map to the same particles.
        from pytorch3d.ops import sample_farthest_points, knn_points
        xyz_static = self.data['gaussians'][:, :3].unsqueeze(0).to(self.device)
        self.patch_centers, _ = sample_farthest_points(xyz_static, K=self.n_patches)
        
        # Calculate KNN interpolation weights for all particles
        init_pos_centered = (self.data['init_pos'].to(self.device) + self.auto_offset).unsqueeze(0)
        dist, self.patch_idx, _ = knn_points(init_pos_centered, self.patch_centers, K=3)
        dist = torch.clamp(dist, min=1e-6)
        inv_dist = 1.0 / dist
        norm = torch.sum(inv_dist, dim=2, keepdim=True)
        self.interp_weights = (inv_dist / norm).unsqueeze(-1)
        
        # 7. Optimizer with Parameter-level Learning Rate Ranges
        lr_ranges = self.cfg.mpm.get('lr_ranges', {})
        base_lr = self.cfg.get('train', {}).get('lr_params', 1e-3)
        
        param_list = [
            {'params': [self.log_weights], 'lr': lr_ranges.get('log_weights', [1e-3])[0]},
            {'params': [self.raw_E], 'lr': lr_ranges.get('E', [base_lr])[0]},
            {'params': [self.raw_nu], 'lr': lr_ranges.get('nu', [base_lr])[0]},
            {'params': [self.raw_fiber_k], 'lr': lr_ranges.get('fiber_k', [base_lr])[0]},
            {'params': [self.raw_fiber_dir], 'lr': lr_ranges.get('fiber_dir', [base_lr])[0]},
            {'params': [self.raw_yield], 'lr': lr_ranges.get('yield_stress', [base_lr])[0]},
            {'params': [self.raw_viscosity], 'lr': lr_ranges.get('plastic_viscosity', [base_lr])[0]},
        ]
        
        # [NEW] Add Residual PGND to optimizer
        if self.residual_net is not None:
            residual_lr = getattr(cfg.residual, 'lr', 1e-4)
            param_list.append({'params': self.residual_net.parameters(), 'lr': residual_lr})
            print(f"[INFO] Added ResidualPGND to optimizer with LR: {residual_lr}")
            
        self.optimizer = optim.Adam(param_list)
        
        # [NEW] Cosine Annealing Scheduler over num_iters
        # We'll initialize it in train() where num_iters is known, or default here
        self.scheduler = None 
        
        # [NEW] Session Iterations Counter for robust Early Stopping
        self.session_iters = 0
        
        # [NEW] Load optimizer/scheduler state if resuming from PT
        if hasattr(self, 'resume_checkpoint') and self.resume_checkpoint is not None:
            # We can't load scheduler yet as it's not inited, but we can load optimizer
            if 'optimizer_state_dict' in self.resume_checkpoint:
                try:
                    self.optimizer.load_state_dict(self.resume_checkpoint['optimizer_state_dict'])
                    print("[RESUME] Optimizer state loaded.")
                except Exception as e:
                    print(f"[RESUME] Warning: Failed to load optimizer state: {e}")

    def get_current_phys_props(self):
        weights = torch.softmax(self.log_weights, dim=-1)
        
        # [STABILITY] Map to a reasonable range from config
        bounds = self.cfg.mpm.get('material_bounds', {})
        
        # [FIXED] Friction is now a constant scalar
        friction = torch.tensor(self.cfg.mpm.boundary.friction, device=self.device)
        
        # [NEW] E, nu -> mu, lam conversion
        E_min, E_max = bounds.get('E', [1e3, 1e6])
        nu_min, nu_max = bounds.get('nu', [0.0, 0.45])
        fk_min, fk_max = bounds.get('fiber_k', [5e3, 5e5])
        ys_min, ys_max = bounds.get('yield_stress', [1e2, 1e5])
        visc_min, visc_max = bounds.get('plastic_viscosity', [0.1, 10.0])

        # [REVISED] Log-space mapping for better optimization stability
        # Map raw parameter (unbounded) to [min, max] using Log-Sigmoid
        # value = exp( log(min) + sigmoid(raw) * (log(max) - log(min)) )
        
        def map_log_sigmoid(raw_param, min_val, max_val):
            log_min = np.log(max(min_val, 1e-6))
            log_max = np.log(max_val)
            log_val = log_min + torch.sigmoid(raw_param) * (log_max - log_min)
            return torch.exp(log_val)
        
        p_E = map_log_sigmoid(self.raw_E, E_min, E_max)
        # Nu is usually small [0, 0.5], keep linear sigmoid for it
        p_nu = nu_min + torch.sigmoid(self.raw_nu) * (nu_max - nu_min)
        
        # E, nu -> mu, lam conversion
        p_mu, p_lam = youngs_poisson_to_lame(p_E, p_nu)
        
        p_fiber_k = map_log_sigmoid(self.raw_fiber_k, fk_min, fk_max)
        p_fiber_dir = self.raw_fiber_dir
        
        # [NEW] Yield Stress & Viscosity for Plasticity
        p_yield = map_log_sigmoid(self.raw_yield, ys_min, ys_max)
        p_visc = map_log_sigmoid(self.raw_viscosity, visc_min, visc_max)
        
        return weights, p_mu.squeeze(), p_lam.squeeze(), p_fiber_k.squeeze(), p_fiber_dir, friction, p_yield.squeeze(), p_E.squeeze(), p_nu.squeeze(), p_visc.squeeze()

    def train(self, num_iters=50):
        print(f"\nStarting MPM Training (System ID) for scene: {self.scene_id}")
        
        # [NEW] Setup Scheduler now that we know num_iters
        lr_ranges = self.cfg.mpm.get('lr_ranges', {})
        # Note: Adam's eta_min is tricky with different groups. 
        # For simplicity, we use CosineAnnealingLR which scales the current LRs down.
        # If we want specific end LRs, we'd need a custom lambda scheduler.
        # Here we scale by (end_lr / start_lr).
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_iters, eta_min=1e-5)
        
        # [NEW] Resume Scheduler State if available
        if hasattr(self, 'resume_checkpoint') and self.resume_checkpoint is not None:
            if 'scheduler_state_dict' in self.resume_checkpoint and self.resume_checkpoint['scheduler_state_dict']:
                try:
                    self.scheduler.load_state_dict(self.resume_checkpoint['scheduler_state_dict'])
                    print("[RESUME] Scheduler state loaded.")
                except Exception as e:
                    print(f"[RESUME] Warning: Failed to load scheduler state: {e}")
            
            # [NEW] Determine start iter
            start_iter = self.resume_checkpoint.get('iter', 0)
            print(f"[RESUME] Resuming from Iter {start_iter}")
        else:
            start_iter = 0
        offset = self.auto_offset
        
        init_pos = (self.data['init_pos'].to(self.device) + offset)
        gt_tracks = (self.data['gt_surface_tracks'].to(self.device) + offset)
        num_supervised = self.data['num_supervised']
        
        # [NEW] Initialize History Buffers for ResidualPGND
        H = getattr(self.cfg.residual if hasattr(self.cfg, 'residual') else None, 'n_history', 2)
        x_history = [] 
        v_history = [] 
        
        T_data = gt_tracks.shape[0]
        T = min(T_data, self.cfg.mpm.max_frames) if self.cfg.mpm.max_frames > 0 else T_data
        
        sim_pos = init_pos.unsqueeze(0)

        # 增加外部进度条
        # [RESUME] Start from correct iter
        last_iter = start_iter
        main_pbar = tqdm(range(start_iter, num_iters), desc=f"[{self.scene_id}] Optimization Progress")

        for i in main_pbar:
            last_iter = i + 1
            self.optimizer.zero_grad()
            self.simulator.reset(init_pos, controller_pos=self.controller_points[0])
            
            # [NEW] Reset history with current initial state
            x_history = [init_pos.clone() for _ in range(H)]
            v_history = [torch.zeros_like(init_pos) for _ in range(H)]
            
            T_data = gt_tracks.shape[0]
            T = min(T_data, self.cfg.mpm.max_frames) if self.cfg.mpm.max_frames > 0 else T_data
            
            # [NEW] Staged Optimization: Handle expert weight and fiber warmup
            # If we are in warmup, freeze weights/fiber. Otherwise, unfreeze.
            weight_warmup = getattr(self.cfg.mpm, 'weight_warmup_iters', 0)
            fiber_warmup = getattr(self.cfg.mpm, 'fiber_warmup_iters', 0)
            
            # [STAGED] Elastic vs Plastic Phases
            # Phase 1 (0% - 50%): Optimize Elastic (E, nu, Weights, Fiber) + Init Weights
            # Phase 2 (50% - 100%): Optimize Plastic (Yield, Viscosity) + Fine-tune everything
            total_iters = num_iters
            phase_split = total_iters // 2
            
            is_elastic_phase = (i < phase_split)
            is_plastic_phase = (i >= phase_split)
            
            # Dynamic Requires Grad Control
            # 1. Weights: Active after weight_warmup
            self.log_weights.requires_grad = (i >= weight_warmup)
            
            # 2. Elastic Params: Always active in Phase 1, optional in Phase 2 (usually keep active for fine-tuning)
            # Here we keep them active throughout to allow joint optimization in Phase 2
            self.raw_E.requires_grad = True 
            self.raw_nu.requires_grad = True
            
            # 3. Fiber Params: Active after fiber_warmup
            self.raw_fiber_dir.requires_grad = (i >= fiber_warmup)
            self.raw_fiber_k.requires_grad = (i >= fiber_warmup)
            
            # 4. Plastic Params: FROZEN in Phase 1, ACTIVE in Phase 2
            self.raw_yield.requires_grad = is_plastic_phase
            self.raw_viscosity.requires_grad = is_plastic_phase
            
            # 5. [REVISED] Residual PGND is ALWAYS active for joint training
            if self.residual_net is not None:
                self.residual_net.requires_grad = True
                self.residual_net.train()

            # [NEW] Indicators for monitoring Residual contribution
            res_stats = {
                'mean_mag': [], 'max_mag': [], 'ratio_to_phys': [], 'cos_sim': []
            }

            total_loss = 0.0
            
            frame_pbar = tqdm(range(T), desc=f"  [{self.scene_id}] Iter {i+1}", leave=False)
            
            def gather_and_interp(patch_data):
                flat_idx = self.patch_idx.squeeze(0).view(-1)
                gathered = patch_data[flat_idx].view(1, -1, 3, patch_data.shape[-1])
                return torch.sum(self.interp_weights * gathered, dim=2).squeeze(0)

            for t in frame_pbar:
                # [MEMORY SAFE] Re-calculate params inside the loop to avoid retain_graph=True.
                w_patch, mu_patch, lam_patch, fk_patch, fdir_patch, friction, yield_patch, E_patch, nu_patch, visc_patch = self.get_current_phys_props()
                
                p_weights = gather_and_interp(w_patch)
                p_mu = gather_and_interp(mu_patch.unsqueeze(-1)).squeeze()
                p_lam = gather_and_interp(lam_patch.unsqueeze(-1)).squeeze()
                p_fk = gather_and_interp(fk_patch.unsqueeze(-1)).squeeze()
                p_fdir = torch.nn.functional.normalize(gather_and_interp(fdir_patch), dim=1, eps=1e-8)
                p_yield = gather_and_interp(yield_patch.unsqueeze(-1)).squeeze()
                p_visc = gather_and_interp(visc_patch.unsqueeze(-1)).squeeze()
                expert_params = {'mu': p_mu, 'lam': p_lam, 'fiber_k': p_fk, 'fiber_dir': p_fdir, 'yield_stress': p_yield, 'plastic_viscosity': p_visc}

                c_pos_end = self.controller_points[t]
                c_pos_start = self.controller_points[t-1] if t > 0 else c_pos_end
                
                v_ctrl_t = (c_pos_end - c_pos_start) / (self.cfg.mpm.dt * self.cfg.mpm.steps_per_frame)

                # Controller Stiffness Warm-up
                orig_stiffness = getattr(self.cfg.mpm, 'controller_stiffness', 1000.0)
                warmup_frames = getattr(self.cfg.mpm, 'controller_warmup_frames', 10)
                current_stiffness = orig_stiffness * min(1.0, (t + 1) / (warmup_frames + 1e-6))
                self.simulator.cfg.controller_stiffness = current_stiffness

                # [SCHEME A] Phase 1: Pure MPM Physics Solver Loop
                # Start of frame position
                x_start_frame = (self.simulator.x - self.simulator.shift).detach().unsqueeze(0)

                for s in range(self.cfg.mpm.steps_per_frame):
                    alpha = (s + 1) / self.cfg.mpm.steps_per_frame
                    curr_target_pos = c_pos_start + alpha * (c_pos_end - c_pos_start)
                    
                    # residual_v is None during sub-steps in Scheme A
                    x_curr = self.simulator.step(p_weights, expert_params, 
                                                 controller_pos=curr_target_pos, 
                                                 controller_vel=v_ctrl_t,
                                                 residual_v=None)

                # [SCHEME A] Phase 2: Neural Feedback Correction
                delta_v = None
                if self.residual_net is not None:
                    # History [B, N, H, 3]
                    x_his_tensor = torch.stack(x_history, dim=1).unsqueeze(0)
                    v_his_tensor = torch.stack(v_history, dim=1).unsqueeze(0)
                    
                    # Current results from physics (Lagrangian)
                    curr_x_mpm = (self.simulator.x - self.simulator.shift).unsqueeze(0)
                    curr_v_mpm = self.simulator.v.unsqueeze(0)
                    
                    # Predict correction: Net(pos_mpm, vel_mpm, pos_start, history)
                    delta_v = self.residual_net(curr_x_mpm, curr_v_mpm, x_start_frame, x_his_tensor, v_his_tensor).squeeze(0)
                    
                    # [NEW] Calculate monitoring metrics before applying
                    with torch.no_grad():
                        v_mpm_mag = torch.norm(curr_v_mpm.squeeze(0), dim=-1)
                        dv_mag = torch.norm(delta_v, dim=-1)
                        
                        res_stats['mean_mag'].append(dv_mag.mean().item())
                        res_stats['max_mag'].append(dv_mag.max().item())
                        
                        # Ratio of correction to physics magnitude
                        ratio = dv_mag / (v_mpm_mag + 1e-6)
                        res_stats['ratio_to_phys'].append(ratio.mean().item())
                        
                        # Cosine similarity (directional alignment)
                        cos_sim = torch.sum(delta_v * curr_v_mpm.squeeze(0), dim=-1) / (dv_mag * v_mpm_mag + 1e-6)
                        res_stats['cos_sim'].append(cos_sim.mean().item())

                    # Apply correction to Simulator State
                    # 1. Correct Velocity
                    self.simulator.v = self.simulator.v + delta_v
                    
                    # 2. Correct Position: x = x + delta_v * (dt_frame)
                    frame_dt = self.cfg.mpm.dt * self.cfg.mpm.steps_per_frame
                    self.simulator.x = self.simulator.x + delta_v * frame_dt
                    
                    # Final x_curr for loss calculation (after correction)
                    x_curr = self.simulator.x - self.simulator.shift

                # Update History Buffer (FIFO)
                if self.residual_net is not None:
                    x_history.pop(0)
                    x_history.append((self.simulator.x - self.simulator.shift).detach())
                    v_history.pop(0)
                    v_history.append(self.simulator.v.detach())
                
                x_curr_surf = x_curr[:num_supervised]
                x_gt = gt_tracks[t]
                
                # [FIXED] Mask out zero-artifacts using RAW GT tracks (before offset)
                # to ensure we catch points at (0,0,0) regardless of auto-centering.
                gt_mask = torch.norm(self.data['gt_surface_tracks'][t], dim=-1) > 1e-5 # [N_surf]
                
                if gt_mask.any():
                    x_curr_masked = x_curr_surf[gt_mask]
                    x_gt_masked = x_gt[gt_mask]
                    
                    track_loss = torch.mean((x_curr_masked - x_gt_masked)**2)
                    cham_loss, _ = chamfer_distance(x_curr_masked.unsqueeze(0), x_gt_masked.unsqueeze(0))
                    
                    # [NEW] Residual Regularization: encourage the network to only make small corrections
                    res_reg = 0.0
                    if delta_v is not None:
                        res_reg = torch.mean(delta_v**2) * getattr(self.cfg.residual, 'lambda_reg', 0.01)
                    
                    frame_loss = (track_loss * 1.0 + cham_loss * 1.0 + res_reg) / T
                else:
                    # Fallback if the whole frame is empty (should not happen)
                    frame_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # 如果出现 NaN/Inf，立即报错并停止
                if not torch.isfinite(frame_loss):
                    print(f"\n[ERROR] Non-finite loss ({frame_loss.item()}) detected in Frame {t}! Stopping.")
                    total_loss = float('nan')
                    break

                # [ULTIMATE STABILITY] Frame-by-frame backward WITHOUT retain_graph.
                # This is the only way to reliably run 666 sub-steps without OOM.
                frame_loss.backward()
                
                # [NEW] Log per-frame loss for debugging stability
                # Using a separate tag for each iteration allows overlaying them in TensorBoard
                # We only log every 1 iteration to see the curve.
                if i % 1 == 0:
                    self.writer.add_scalar(f'Frame_Loss_Detail/Iter_{i:03d}', frame_loss.item() * T, t)

                # [NEW] 3D Point Cloud Visualization in TensorBoard (MESH Tab)
                # We log every iter, and every 10 frames within that iter for detailed monitoring.
                if i % 1 == 0 and t % 10 == 0:
                    with torch.no_grad():
                        v_sim = x_curr.detach()
                        v_gt = x_gt.detach()
                        # Use same mask to only show valid GT points
                        gt_mask_viz = torch.norm(self.data['gt_surface_tracks'][t], dim=-1).to(self.device) > 1e-5
                        v_gt = v_gt[gt_mask_viz]
                        v_ctrl = curr_target_pos.detach()
                        
                        # Concatenate all points for one-shot logging
                        all_v = torch.cat([v_sim, v_gt, v_ctrl], dim=0).unsqueeze(0).cpu() # [1, N, 3]
                        
                        # Define colors (RGB: 0-255)
                        c_sim = torch.tensor([[0, 0, 255]], device='cpu').repeat(v_sim.shape[0], 1)
                        c_gt = torch.tensor([[0, 255, 0]], device='cpu').repeat(v_gt.shape[0], 1)
                        c_ctrl = torch.tensor([[255, 0, 0]], device='cpu').repeat(v_ctrl.shape[0], 1)
                        all_c = torch.cat([c_sim, c_gt, c_ctrl], dim=0).unsqueeze(0) # [1, N, 3]
                        
                        # [UPDATED] Use scene_id as top-level tag to force separation in TensorBoard
                        self.writer.add_mesh(f'{self.scene_id}_Mesh/Iter_{i:03d}', vertices=all_v, colors=all_c, global_step=t)

                # Detach simulator states to free simulation graph
                self.simulator.x = self.simulator.x.detach().requires_grad_()
                self.simulator.v = self.simulator.v.detach().requires_grad_()
                self.simulator.F = self.simulator.F.detach().requires_grad_()
                self.simulator.C = self.simulator.C.detach().requires_grad_()
                
                total_loss += frame_loss.item()
                
                # 实时监控 E 和 nu
                frame_pbar.set_postfix({
                    'f_loss': f"{frame_loss.item() * T:.4f}",
                    'E': f"{E_patch.mean().item():.1e}",
                    'nu': f"{nu_patch.mean().item():.3f}"
                })


            # --- Gradient Clipping ---
            # [STABILITY] Include ALL learnable parameters (Physics + Neural) in clipping
            all_params = [self.log_weights, self.raw_E, self.raw_nu, self.raw_fiber_k, self.raw_fiber_dir, self.raw_yield, self.raw_viscosity]
            if self.residual_net is not None:
                all_params.extend(list(self.residual_net.parameters()))
                
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.1)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm == 0:
                print(f"\n[WARNING] Invalid gradient norm ({grad_norm}) in Iter {i+1}! Skipping.")
                self.optimizer.zero_grad()
            else:
                # [DEBUG] 使用本次 Iter 最后一帧计算的物理参数进行日志记录
                with torch.no_grad():
                    w_mean = w_patch.mean(dim=0).tolist()
                    w_str = ", ".join([f"{self.active_experts[j]}:{w_mean[j]:.3f}" for j in range(len(self.active_experts))])
                    print(f"\n[DEBUG] Iter {i+1} Grad Norm: {grad_norm:.2e}")
                    print(f"        Mean E: {E_patch.mean().item():.2e}, Mean nu: {nu_patch.mean().item():.3f}")
                    print(f"        Mean Mu: {mu_patch.mean().item():.2f}, Mean Lam: {lam_patch.mean().item():.2f}")
                    print(f"        Mean Weights: [{w_str}]")
                self.optimizer.step()
                
                # [FIXED] Only step scheduler if the optimizer actually stepped
                if self.scheduler is not None:
                    self.scheduler.step()
                
                for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f'LR/Group_{group_idx}', param_group['lr'], i)

            total_loss_val = total_loss if isinstance(total_loss, float) else total_loss.item()
            current_iter_loss = total_loss_val * T
            main_pbar.set_postfix({'total_loss': f"{current_iter_loss:.6f}"})
            
            self.writer.add_scalar('Loss/Total', current_iter_loss, i)
            
            # --- [NEW] Early Stopping Check ---
            if not hasattr(self, 'loss_history'):
                self.loss_history = []
            self.loss_history.append(current_iter_loss)
            
            # [FIXED] Increment session_iters to ensure we don't immediately early-stop on resume
            self.session_iters += 1
            
            patience = getattr(self.cfg.train, 'early_stop_patience', 10)
            min_delta = getattr(self.cfg.train, 'early_stop_min_delta', 1e-4)
            
            # Logic: Only check early stopping if we have enough total history AND 
            # we have run for at least 'patience' iterations in the CURRENT session.
            if len(self.loss_history) > patience and self.session_iters > patience:
                # 检查过去 patience 次迭代中，最好的 loss 相比之前的改进
                recent_best = min(self.loss_history[-patience:])
                prev_best = min(self.loss_history[:-patience])
                
                if (prev_best - recent_best) < min_delta:
                    print(f"\n[EARLY STOP] No significant improvement for {patience} iters in this session. Best loss: {recent_best:.6f}. Stopping.")
                    break

            # --- [NEW] 记录残差模块的贡献指标 ---
            if self.residual_net is not None and len(res_stats['mean_mag']) > 0:
                self.writer.add_scalar('Residual/Mean_Magnitude', np.mean(res_stats['mean_mag']), i)
                self.writer.add_scalar('Residual/Max_Magnitude', np.mean(res_stats['max_mag']), i)
                self.writer.add_scalar('Residual/Correction_to_Physics_Ratio', np.mean(res_stats['ratio_to_phys']), i)
                self.writer.add_scalar('Residual/Cosine_Similarity', np.mean(res_stats['cos_sim']), i)
                
                # 记录最后一次计算的 delta_v 直方图
                self.writer.add_histogram('Residual/Delta_V_Distribution', delta_v.detach().cpu().numpy(), i)

            # 记录平均参数到 TensorBoard
            self.writer.add_scalar('Params/E_Mean', E_patch.mean().item(), i)
            self.writer.add_scalar('Params/Nu_Mean', nu_patch.mean().item(), i)
            self.writer.add_scalar('Params/Mu_Mean', mu_patch.mean().item(), i)
            self.writer.add_scalar('Params/Lam_Mean', lam_patch.mean().item(), i)
            self.writer.add_scalar('Params/Global_Friction', friction.item(), i)
            self.writer.add_scalar('Params/Yield_Stress_Mean', yield_patch.mean().item(), i)
            self.writer.add_scalar('Params/Viscosity_Mean', visc_patch.mean().item(), i)
            
            # --- [NEW] 记录更多优化参数 ---
            # 1. 记录本构专家权重 (Weights)
            w_mean = w_patch.mean(dim=0)
            for e_idx, expert_name in enumerate(self.active_experts):
                self.writer.add_scalar(f'Weights/{expert_name}', w_mean[e_idx].item(), i)
            
            # 2. 记录纤维强度 (Fiber K)
            if 'fi' in self.active_experts:
                self.writer.add_scalar('Params/Fiber_K_Mean', fk_patch.mean().item(), i)
            
            # 3. 记录梯度范数 (Grad Norm)，观察优化稳定性
            self.writer.add_scalar('Train/Grad_Norm', grad_norm, i)

            # --- [NEW] Save Best Checkpoint ---
            current_total_loss = total_loss_val * T
            if current_total_loss < self.best_loss:
                self.best_loss = current_total_loss
                best_path = os.path.join(self.cfg.output_dir, self.scene_id, "best_checkpoint.pt")
                self.save_checkpoint(best_path, iter=i+1)
                # Also save the results pkl for convenience
                self.save_results(os.path.join(self.cfg.output_dir, self.scene_id, "best_optimized_params.pkl"))
                print(f"[BEST] New best loss: {self.best_loss:.6f} at Iter {i+1}. Saved to {best_path}")

            if (i+1) % 5 == 0:
                self.save_results(os.path.join(self.log_dir, f"params_iter_{i+1}.pkl"))
                # [NEW] Save full checkpoint for resumption
                self.save_checkpoint(os.path.join(self.log_dir, f"checkpoint_iter_{i+1}.pt"), iter=i+1)

        final_path = os.path.join(self.cfg.output_dir, self.scene_id, "optimized_params.pkl")
        self.save_results(final_path)
        # [NEW] Save final checkpoint
        self.save_checkpoint(os.path.join(self.cfg.output_dir, self.scene_id, "final_checkpoint.pt"), iter=last_iter)
        
        print(f"MPM Training Finished! Saved to {final_path}")
        
        # --- [NEW] Qualitative Visualization ---
        print(f"Generating visualization video for {self.scene_id}...")
        video_path = os.path.join(self.cfg.output_dir, self.scene_id, "final_simulation.mp4")
        self.visualize(video_path)
        print(f"Visualization video saved to {video_path}")

    def load_from_checkpoint(self, resume_path):
        print(f"[RESUME] Loading state from {resume_path}...")
        
        if resume_path.endswith('.pt'):
            # Load Full Checkpoint
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            state = checkpoint['model_state_dict']
            with torch.no_grad():
                self.log_weights.copy_(state['log_weights'])
                self.raw_E.copy_(state['raw_E'])
                self.raw_nu.copy_(state['raw_nu'])
                self.raw_fiber_k.copy_(state['raw_fiber_k'])
                self.raw_fiber_dir.copy_(state['raw_fiber_dir'])
                self.raw_yield.copy_(state['raw_yield'])
                self.raw_viscosity.copy_(state['raw_viscosity'])
                
                # [NEW] Restore Best Loss and Loss History
                if 'best_loss' in checkpoint:
                    self.best_loss = checkpoint['best_loss']
                    print(f"[RESUME] Best loss restored: {self.best_loss:.6f}")
                if 'loss_history' in checkpoint:
                    self.loss_history = checkpoint['loss_history']
                    print(f"[RESUME] Loss history restored ({len(self.loss_history)} iters).")

                # [NEW] Restore Auto-centering Offset if available
                if 'auto_offset' in state:
                    self.auto_offset = state['auto_offset'].to(self.device)
                    # Sync to simulator
                    if hasattr(self, 'simulator'):
                        self.simulator.base_offset = self.auto_offset
                        self.simulator._apply_boundary()
                    print("[RESUME] Auto-centering offset restored.")

                # [NEW] Restore Patch Centers and Interpolation Weights if available
                # This is CRITICAL for consistent parameter mapping
                if 'patch_centers' in state:
                    self.patch_centers = state['patch_centers'].to(self.device)
                    from pytorch3d.ops import knn_points
                    init_pos_centered = (self.data['init_pos'].to(self.device) + self.auto_offset).unsqueeze(0)
                    dist, self.patch_idx, _ = knn_points(init_pos_centered, self.patch_centers, K=3)
                    dist = torch.clamp(dist, min=1e-6)
                    inv_dist = 1.0 / dist
                    norm = torch.sum(inv_dist, dim=2, keepdim=True)
                    self.interp_weights = (inv_dist / norm).unsqueeze(-1)
                    print("[RESUME] Persistent patch assignment restored.")

                # [NEW] Load ResidualPGND weights if available
                if 'residual_net' in state and self.residual_net is not None:
                    try:
                        self.residual_net.load_state_dict(state['residual_net'])
                        print("[RESUME] ResidualPGND weights loaded.")
                    except Exception as e:
                        print(f"[RESUME] Warning: Failed to load ResidualPGND weights: {e}")
            
            # We need to defer optimizer/scheduler load until train() creates them
            self.resume_checkpoint = checkpoint 
        else:
            # Load Legacy PKL
            with open(resume_path, 'rb') as f:
                ckpt = pickle.load(f)
            
            with torch.no_grad():
                if 'raw_E' in ckpt: self.raw_E.copy_(torch.from_numpy(ckpt['raw_E']).to(self.device))
                if 'raw_nu' in ckpt: self.raw_nu.copy_(torch.from_numpy(ckpt['raw_nu']).to(self.device))
                if 'raw_fiber_k' in ckpt: self.raw_fiber_k.copy_(torch.from_numpy(ckpt['raw_fiber_k']).to(self.device))
                if 'raw_fiber_dir' in ckpt: self.raw_fiber_dir.copy_(torch.from_numpy(ckpt['raw_fiber_dir']).to(self.device))
                if 'raw_yield' in ckpt: self.raw_yield.copy_(torch.from_numpy(ckpt['raw_yield']).to(self.device))
                if 'raw_viscosity' in ckpt: self.raw_viscosity.copy_(torch.from_numpy(ckpt['raw_viscosity']).to(self.device))
                if 'log_weights' in ckpt: self.log_weights.copy_(torch.from_numpy(ckpt['log_weights']).to(self.device))
            self.resume_checkpoint = None

    def test(self, checkpoint_path, output_path=None):
        """
        Load a specific checkpoint and run visualization/inference.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint {checkpoint_path} does not exist.")
            return

        self.load_from_checkpoint(checkpoint_path)
        
        if output_path is None:
             base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
             output_path = os.path.join(os.path.dirname(checkpoint_path), f"{base_name}_test.mp4")
             
        print(f"[TEST] Running inference with checkpoint: {checkpoint_path}")
        self.visualize(output_path)
        print(f"[TEST] Result saved to {output_path}")

    def visualize(self, output_path):
        """
        Run one final simulation and save as a video.
        """
        # 1. Setup for final run
        self.simulator.eval()
        
        # Use automatically calculated offset
        offset = self.auto_offset
        
        init_pos = (self.data['init_pos'].to(self.device) + offset)
        controller_points = self.controller_points
        gt_tracks = (self.data['gt_surface_tracks'].to(self.device) + offset)
        num_supervised = self.data['num_supervised']
        
        # Get optimized props
        w_patch, mu_patch, lam_patch, fk_patch, fdir_patch, friction, yield_patch, E_patch, nu_patch, visc_patch = self.get_current_phys_props()
        
        def gather_and_interp(patch_data):
            flat_idx = self.patch_idx.squeeze(0).view(-1)
            gathered = patch_data[flat_idx].view(1, -1, 3, patch_data.shape[-1])
            return torch.sum(self.interp_weights * gathered, dim=2).squeeze(0)

        p_weights = gather_and_interp(w_patch)
        p_mu = gather_and_interp(mu_patch.unsqueeze(-1)).squeeze()
        p_lam = gather_and_interp(lam_patch.unsqueeze(-1)).squeeze()
        p_fk = gather_and_interp(fk_patch.unsqueeze(-1)).squeeze()
        p_fdir = torch.nn.functional.normalize(gather_and_interp(fdir_patch), dim=1, eps=1e-8)
        p_yield = gather_and_interp(yield_patch.unsqueeze(-1)).squeeze()
        p_visc = gather_and_interp(visc_patch.unsqueeze(-1)).squeeze()
        expert_params = {'mu': p_mu, 'lam': p_lam, 'fiber_k': p_fk, 'fiber_dir': p_fdir, 'yield_stress': p_yield, 'plastic_viscosity': p_visc}

        # 2. Run Simulation
        self.simulator.reset(init_pos, controller_pos=controller_points[0])
        
        # [NEW] Initialize History Buffers for ResidualPGND
        H = getattr(self.cfg.residual if hasattr(self.cfg, 'residual') else None, 'n_history', 2)
        x_history = [init_pos.clone() for _ in range(H)]
        v_history = [torch.zeros_like(init_pos) for _ in range(H)]

        # Note: self.simulator.current_friction is already set from config via __init__
        T_data = controller_points.shape[0]
        T = min(T_data, self.cfg.mpm.max_frames) if self.cfg.mpm.max_frames > 0 else T_data
        
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        frames = []
        with torch.no_grad():
            for t in tqdm(range(T), desc="Rendering Frames"):
                c_pos_end = controller_points[t]
                c_pos_start = controller_points[t-1] if t > 0 else c_pos_end
                
                v_ctrl_t = (c_pos_end - c_pos_start) / (self.cfg.mpm.dt * self.cfg.mpm.steps_per_frame)

                # [SCHEME A] Phase 1: Pure MPM Physics Solver Loop
                # Start of frame position
                x_start_frame = (self.simulator.x - self.simulator.shift).detach().unsqueeze(0)

                for s in range(self.cfg.mpm.steps_per_frame):
                    alpha = (s + 1) / self.cfg.mpm.steps_per_frame
                    curr_target_pos = c_pos_start + alpha * (c_pos_end - c_pos_start)
                    
                    # residual_v is None during sub-steps in Scheme A
                    x_curr = self.simulator.step(p_weights, expert_params, 
                                                 controller_pos=curr_target_pos, 
                                                 controller_vel=v_ctrl_t,
                                                 residual_v=None)
                
                # [SCHEME A] Phase 2: Neural Feedback Correction
                if self.residual_net is not None:
                    # Set to eval mode just in case
                    self.residual_net.eval()
                    
                    # History [B, N, H, 3]
                    x_his_tensor = torch.stack(x_history, dim=1).unsqueeze(0)
                    v_his_tensor = torch.stack(v_history, dim=1).unsqueeze(0)
                    
                    # Current results from physics (Lagrangian)
                    curr_x_mpm = (self.simulator.x - self.simulator.shift).unsqueeze(0)
                    curr_v_mpm = self.simulator.v.unsqueeze(0)
                    
                    # Predict correction
                    delta_v = self.residual_net(curr_x_mpm, curr_v_mpm, x_start_frame, x_his_tensor, v_his_tensor).squeeze(0)
                    
                    # Apply correction to Simulator State
                    self.simulator.v = self.simulator.v + delta_v
                    frame_dt = self.cfg.mpm.dt * self.cfg.mpm.steps_per_frame
                    self.simulator.x = self.simulator.x + delta_v * frame_dt
                    
                    # Update x_curr for plotting
                    x_curr = self.simulator.x - self.simulator.shift

                    # Update History Buffer (FIFO)
                    x_history.pop(0)
                    x_history.append((self.simulator.x - self.simulator.shift).detach())
                    v_history.pop(0)
                    v_history.append(self.simulator.v.detach())

                # Ensure memory is freed between frames
                self.simulator.x = self.simulator.x.detach()
                self.simulator.v = self.simulator.v.detach()
                self.simulator.F = self.simulator.F.detach()
                self.simulator.C = self.simulator.C.detach()
                
                # Plot frame
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Object particles
                obj_x = (x_curr.detach() - offset).cpu().numpy()
                ax.scatter(obj_x[:, 0], obj_x[:, 1], obj_x[:, 2], s=1, c='blue', alpha=0.5, label='Simulation')
                
                # Ground Truth particles (Surface tracks)
                gt_x_raw = (gt_tracks[t].detach() - offset)
                # [FIXED] Mask out zero-artifacts from GT visualization using RAW tracks
                gt_mask = torch.norm(self.data['gt_surface_tracks'][t], dim=-1) > 1e-5
                gt_x = gt_x_raw[gt_mask].cpu().numpy()
                ax.scatter(gt_x[:, 0], gt_x[:, 1], gt_x[:, 2], s=1, c='green', alpha=0.3, label='Ground Truth')
                
                # Controller points
                ctrl_x = (c_pos_end.detach() - offset).cpu().numpy()
                if ctrl_x.shape[0] > 0:
                    ax.scatter(ctrl_x[:, 0], ctrl_x[:, 1], ctrl_x[:, 2], s=20, c='red', marker='x', label='Controller')
                    
                    # [NEW] Draw connections (if available) to show grip
                    # We need access to simulator.controller_indices and simulator.num_ctrl_points
                    if hasattr(self.simulator, 'controller_indices') and self.simulator.controller_indices is not None:
                         # Indices are flattened: [N_ctrl * K]
                        indices = self.simulator.controller_indices.view(-1).cpu().numpy()
                        # Connected object particles: [N_ctrl * K, 3]
                        conn_obj_x = obj_x[indices]
                        
                        # Controller points repeated: [N_ctrl, 3] -> [N_ctrl * K, 3]
                        K = getattr(self.cfg.mpm, 'controller_max_neighbors', 16)
                        # Be careful: actual neighbors per point might be less if filtered, but here we used KNN with fixed K
                        # However, our indices are [N_ctrl, K], so we can just repeat_interleave
                        conn_ctrl_x = np.repeat(ctrl_x, K, axis=0)
                        
                        # We also need the mask to only draw valid connections
                        if hasattr(self.simulator, 'controller_mask'):
                            mask = self.simulator.controller_mask.view(-1).cpu().numpy()
                            # Filter
                            valid_obj = conn_obj_x[mask]
                            valid_ctrl = conn_ctrl_x[mask]
                            
                            # Draw lines. To be fast, we can't do ax.plot for each line.
                            # We can plot a single line with NaNs to break segments.
                            # Format: x1, x2, nan, x3, x4, nan ...
                            n_lines = valid_obj.shape[0]
                            # Limit number of lines to avoid cluttering if too many
                            if n_lines > 500:
                                step = n_lines // 500
                                valid_obj = valid_obj[::step]
                                valid_ctrl = valid_ctrl[::step]
                                n_lines = valid_obj.shape[0]
                                
                            line_x = np.empty(n_lines * 3)
                            line_y = np.empty(n_lines * 3)
                            line_z = np.empty(n_lines * 3)
                            
                            line_x[0::3] = valid_ctrl[:, 0]
                            line_x[1::3] = valid_obj[:, 0]
                            line_x[2::3] = np.nan
                            
                            line_y[0::3] = valid_ctrl[:, 1]
                            line_y[1::3] = valid_obj[:, 1]
                            line_y[2::3] = np.nan
                            
                            line_z[0::3] = valid_ctrl[:, 2]
                            line_z[1::3] = valid_obj[:, 2]
                            line_z[2::3] = np.nan
                            
                            ax.plot(line_x, line_y, line_z, color='red', alpha=0.2, linewidth=0.5)
                
                # Fixed bounds for consistent video
                ax.set_xlim([-0.5, 0.5])
                ax.set_ylim([-0.5, 0.5])
                ax.set_zlim([-0.5, 0.5])
                ax.set_axis_off()
                ax.grid(False)
                # ax.set_title(f"Frame {t}")
                ax.legend(loc='upper right')
                
                frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
                plt.savefig(frame_path)
                plt.close(fig)
                frames.append(frame_path)

        # 3. Synthesize Video with System FFmpeg
        print(f"Synthesizing video to {output_path}...")
        input_pattern = os.path.abspath(os.path.join(temp_dir, 'frame_%04d.png'))
        output_abs_path = os.path.abspath(output_path)
        
        # Use absolute path to system ffmpeg to bypass conda environment's limited version
        ffmpeg_bin = "/usr/bin/ffmpeg"
        if not os.path.exists(ffmpeg_bin):
            ffmpeg_bin = "ffmpeg" # Fallback
            
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
            success = os.path.exists(output_abs_path)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed with exit code {e.returncode}. See above for details.")
            success = False
        
        # 4. Cleanup
        if success:
            shutil.rmtree(temp_dir)
        else:
            print(f"Keeping temp frames at {temp_dir} for debugging.")

    def save_checkpoint(self, path, iter):
        """
        Save full training state for resumption (Optimizer, Scheduler, etc.)
        """
        model_state = {
            'log_weights': self.log_weights,
            'raw_E': self.raw_E,
            'raw_nu': self.raw_nu,
            'raw_fiber_k': self.raw_fiber_k,
            'raw_fiber_dir': self.raw_fiber_dir,
            'raw_yield': self.raw_yield,
            'raw_viscosity': self.raw_viscosity,
            'patch_centers': self.patch_centers,
            'auto_offset': self.auto_offset # [NEW] Save auto-centering offset
        }
        
        # [NEW] Include ResidualPGND weights if available
        if self.residual_net is not None:
            model_state['residual_net'] = self.residual_net.state_dict()

        checkpoint = {
            'iter': iter,
            'best_loss': self.best_loss,      # [NEW] Save best loss info
            'loss_history': getattr(self, 'loss_history', []), # [NEW] Save loss history for early stopping
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }
        torch.save(checkpoint, path)
        print(f"[CHECKPOINT] Full training state saved to {path}")

    def save_results(self, path):
        w, mu, lam, fk, fdir, friction, ys, E, nu, visc = self.get_current_phys_props()
        data = {
            'scene_id': self.scene_id,
            'weights': w.detach().cpu().numpy(),
            'mu': mu.detach().cpu().numpy(),
            'lam': lam.detach().cpu().numpy(),
            'fiber_k': fk.detach().cpu().numpy(),
            'fiber_dir': fdir.detach().cpu().numpy(),
            'friction': friction.item(),
            'yield_stress': ys.detach().cpu().numpy(),
            'plastic_viscosity': visc.detach().cpu().numpy(),
            # Optimized values for reference
            'E': E.detach().cpu().numpy(),
            'nu': nu.detach().cpu().numpy(),
            # Raw learnable parameters for resuming
            'raw_E': self.raw_E.detach().cpu().numpy(),
            'raw_nu': self.raw_nu.detach().cpu().numpy(),
            'raw_fiber_k': self.raw_fiber_k.detach().cpu().numpy(),
            'raw_fiber_dir': self.raw_fiber_dir.detach().cpu().numpy(),
            'raw_yield': self.raw_yield.detach().cpu().numpy(),
            'raw_viscosity': self.raw_viscosity.detach().cpu().numpy(),
            'log_weights': self.log_weights.detach().cpu().numpy()
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
