
import torch
import torch.nn as nn
from mpm_pytorch import MPMSolver, set_boundary_conditions
from ..model.experts.mixture_model import MixtureElasticity, HenckyViscoPlasticity
from pytorch3d.ops import knn_points

class MPMSimulator(nn.Module):
    """
    Wrapper around mpm_pytorch.MPMSolver to support Mixture-of-Experts
    and Controller Interaction (via soft spring coupling).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.base_offset = None  # Store auto-centering offset from trainer
        self.current_friction = getattr(cfg.boundary, 'friction', 0.3) # [NEW] Support dynamic friction
        
        # Initialize the Mixture Constitutive Model with active experts from config
        active_experts = getattr(cfg, 'active_experts', ['nh', 'co', 'st', 'fi'])
        svd_clamp_max = getattr(cfg, 'svd_clamp_max', 2.0)
        self.mixture_model = MixtureElasticity(active_experts=active_experts, svd_clamp_max=svd_clamp_max)
        self.plasticity_model = HenckyViscoPlasticity() # [NEW] Formal Plasticity Component
        
        # Initialize mpm_pytorch solver
        grid_damping = getattr(cfg, 'grid_v_damping_scale', 1.0)
        self.mpm_solver = MPMSolver(
            init_pos=torch.zeros((cfg.n_particles, 3), device=self.device), # Placeholder
            num_grids=cfg.grid_res,
            dt=cfg.dt,
            gravity=getattr(cfg, 'gravity', [0.0, 0.0, -9.8]),
            damping=grid_damping, # [NEW] Apply grid velocity damping
            device=self.device
        )
        
        # Boundary conditions
        self._apply_boundary()

        # Controller Connection State
        self.controller_indices = None # [N_ctrl, K] indices of object particles connected to hands
        self.num_ctrl_points = 0
        self.initial_offsets = None    # [N_ctrl, K, 3] initial distance to maintain

    def _apply_boundary(self):
        boundary_cfg = self.cfg.boundary
        # Shift boundary point if it exists
        point = list(boundary_cfg.point)
        if hasattr(self, 'shift'):
            point = [p + s for p, s in zip(point, self.shift.tolist())]
        
        # [STABILITY] Apply auto-centering offset to boundary as well
        # to maintain relative distance between object and ground.
        if self.base_offset is not None:
            point = [p + o for p, o in zip(point, self.base_offset.tolist())]
            
        bc_params = [
            {
                'type': 'surface_collider',
                'point': point,
                'normal': list(boundary_cfg.normal),
                'surface': boundary_cfg.surface,
                'friction': self.current_friction, # Use dynamic friction
                'restitution': getattr(boundary_cfg, 'restitution', 0.0),
                'start_time': boundary_cfg.start_time,
                'end_time': boundary_cfg.end_time
            }
        ]
        set_boundary_conditions(self.mpm_solver, bc_params)

    def reset(self, init_particles, controller_pos=None):
        """
        Reset solver state with new particles and optionally establish controller connections.
        Args:
            init_particles: [N_sim, 3] positions (original coordinate system)
            controller_pos: [N_ctrl, 3] controller positions at frame 0
        """
        # Shift particles to the center of the MPM grid [0, 1]^3
        self.shift = torch.tensor([0.5, 0.5, 0.5], device=self.device)
        shifted_particles = init_particles + self.shift
        
        self.mpm_solver = MPMSolver(
            init_pos=shifted_particles,
            num_grids=self.cfg.grid_res,
            dt=self.cfg.dt,
            gravity=getattr(self.cfg, 'gravity', [0.0, 0.0, -9.8]),
            device=self.device
        )
        self._apply_boundary()

        self.x = shifted_particles.clone()
        self.v = torch.zeros_like(shifted_particles)
        self.F = torch.eye(3, device=self.device).repeat(shifted_particles.shape[0], 1, 1)
        self.C = torch.zeros((shifted_particles.shape[0], 3, 3), device=self.device)

        # Handle Controller Initialization (Soft Coupling)
        if controller_pos is not None and controller_pos.shape[0] > 0:
            shifted_controller = controller_pos + self.shift
            
            # [FIXED] Use all controller points, removed distance threshold filtering
            active_controller = shifted_controller
            self.num_ctrl_points = active_controller.shape[0]
            
            # Find nearest object particles to ALL controller points
            dist, idx, _ = knn_points(
                active_controller.unsqueeze(0), 
                self.x.unsqueeze(0), 
                K=getattr(self.cfg, 'controller_max_neighbors', 16)
            )
            
            # Filter by radius if provided
            radius = getattr(self.cfg, 'controller_radius', 0.05)
            mask = dist.sqrt() < radius 
            
            self.controller_indices = idx.squeeze(0) 
            self.controller_mask = mask.squeeze(0)   
            
            # [FIX] Store initial relative offsets
            p_idx = self.controller_indices.view(-1)
            connected_x = self.x[p_idx].view(self.num_ctrl_points, -1, 3)
            self.initial_offsets = (active_controller.unsqueeze(1) - connected_x).detach()
            
            # [NEW] Pre-calculate connection counts for normalization in step()
            # count_per_particle[i] = how many controller points are pulling particle i
            self.count_per_particle = torch.zeros(self.x.shape[0], 1, device=self.device)
            self.count_per_particle.index_add_(0, self.controller_indices.view(-1), self.controller_mask.view(-1, 1).float())
            self.count_per_particle = torch.clamp(self.count_per_particle, min=1.0) # Avoid div by zero
            
            self.num_connections = self.controller_mask.sum().item()
        else:
            self.controller_indices = None
            self.num_ctrl_points = 0
            self.initial_offsets = None
            self.num_connections = 0

    def step(self, expert_weights, expert_params, controller_pos=None, controller_vel=None, residual_v=None):
        """
        Execute one MPM step with optional controller forces and neural residuals.
        Args:
            expert_weights: [N_sim, 4]
            expert_params: dict
            controller_pos: [N_ctrl, 3] current controller positions (original coordinate system)
            controller_vel: [N_ctrl, 3] current controller velocities (original coordinate system)
            residual_v: [N_sim, 3] (Optional) velocity residual from PGND
        """
        # 1. Apply Controller PD Forces (before MPM update)
        if self.controller_indices is not None and controller_pos is not None:
            # [FIXED] Simplified position and velocity handling (all points are active)
            active_controller_pos = controller_pos
            shifted_controller = active_controller_pos + self.shift
            
            v_ctrl = controller_vel if controller_vel is not None else torch.zeros_like(shifted_controller)
            
            # Gather positions and velocities of connected particles
            p_idx = self.controller_indices.view(-1)
            connected_x = self.x[p_idx].view(self.num_ctrl_points, -1, 3)
            connected_v = self.v[p_idx].view(self.num_ctrl_points, -1, 3)
            
            # PD Force logic
            target_pos = shifted_controller.unsqueeze(1)
            target_vel = v_ctrl.unsqueeze(1)
            current_offset = (target_pos - connected_x)
            
            kp = getattr(self.cfg, 'controller_stiffness', 1000.0)
            kd = getattr(self.cfg, 'controller_damping', 0.0)
            
            accel = kp * (current_offset - self.initial_offsets) + kd * (target_vel - connected_v)
            accel = accel * self.controller_mask.unsqueeze(-1)
            
            flat_accel = accel.view(-1, 3)
            flat_indices = self.controller_indices.view(-1)
            dv = flat_accel * self.cfg.dt
            
            total_dv = torch.zeros_like(self.v)
            total_dv.index_add_(0, flat_indices, dv)
            
            # [REMOVED] Force Normalization: Cancelled to boost gradient signal and interaction strength.
            # total_dv = total_dv / self.count_per_particle
            
            # [REVISED] Force Normalization: 
            # Use a soft normalization to prevent force stacking while keeping signal strong.
            total_dv = total_dv / torch.pow(self.count_per_particle, 0.5)
            
            # [STABILITY] Clamp total dv - Headroom controlled by config
            clamp_val = getattr(self.cfg, 'controller_clamp_dv', 2.0)
            hit_mask = (total_dv.abs() >= (clamp_val - 1e-4))
            num_clamped = hit_mask.any(dim=-1).sum().item()
            if num_clamped > 0 and getattr(self, 'debug_mode', False):
                max_dv = total_dv.abs().max().item()
                # print(f"[DEBUG] Velocity Clamping: {num_clamped}/{self.x.shape[0]} particles hit limit. Max attempted dV: {max_dv:.4f}")
            
            total_dv = torch.clamp(total_dv, min=-clamp_val, max=clamp_val)
            
            self.v = self.v + total_dv

        # 2. Compute Stress
        params = {
            'weights': expert_weights,
            'mu': expert_params['mu'],
            'lam': expert_params['lam'],
            'fiber_k': expert_params['fiber_k'],
            'fiber_dir': expert_params['fiber_dir'],
            'yield_stress': expert_params.get('yield_stress'),
            'plastic_viscosity': expert_params.get('plastic_viscosity')
        }
        self.mixture_model.current_params = params
        stress = self.mixture_model(self.F)
        
        # [STABILITY] Stress Protection: Zero out NaNs/Infs if any expert leaked them
        if not torch.isfinite(stress).all():
            stress = torch.where(torch.isfinite(stress), stress, torch.zeros_like(stress))
        
        # 3. Solver Update
        self.x, self.v, self.C, self.F = self.mpm_solver(self.x, self.v, self.C, self.F, stress)
        
        # 4. Inject Neural Residual (PGND)
        if residual_v is not None:
            # The residual is added to the velocity before plasticity and integration
            self.v = self.v + residual_v
        
        # 5. [NEW] Global Plasticity Mapping (Implicit Return Mapping)
        # Limits the deviatoric part of the logarithmic strain for stability and realism.
        if expert_params.get('yield_stress') is not None:
            self.F = self.plasticity_model(
                self.F, 
                expert_params['mu'], 
                expert_params['yield_stress'], 
                expert_params.get('plastic_viscosity', torch.ones_like(expert_params['mu'])),
                self.cfg.dt
            )
        
        # --- Post-Update Adjustments: Damping & Clipping ---
        damping = getattr(self.cfg, 'damping', 0.0)
        if damping > 0:
            self.v = self.v * (1.0 - damping * self.cfg.dt)
            
        # Ensure particles stay within boundaries
        # Note: self.x is shifted [0.5, 0.5, 0.5], keep it within [0.1, 0.9] to avoid grid edges
        self.x = torch.clamp(self.x, min=0.05, max=0.95)
            
        # --- Stability Guard: Velocity Clipping ---
        # strictly limited to 10.0 for safety (ensures stability in explicit integration)
        self.v = torch.clamp(self.v, min=-10.0, max=10.0) 
        
        return self.x - self.shift

