"""Test gradient flow through Warp MPM kernels"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import warp as wp
wp.init()

import sys
sys.path.insert(0, '.')
from phys_expert.model.diff_simulator.warp_solver.mpm_solver_warp import MPM_Simulator_WARP
from phys_expert.model.diff_simulator.warp_solver.warp_utils import torch2warp_vec3, torch2warp_float

N = 100
device = 'cuda:0'

solver = MPM_Simulator_WARP(N, n_grid=64, device=device)

# Create initial particles in a small cube centered in the grid
x_init = 0.5 + 0.05 * (torch.rand(N, 3, device=device) - 0.5)
vol = torch.ones(N, device=device) * (1.0/64)**3

solver.load_initial_data_from_torch(x_init, vol, n_grid=64, grid_lim=1.0, device=device)

# FIX: Ensure particle_x has requires_grad=True for gradient flow
solver.mpm_state.particle_x.requires_grad = True

# Check requires_grad status
s = solver.mpm_state
print("=== requires_grad status after load_initial_data_from_torch ===")
print(f"  particle_x: req_grad={s.particle_x.requires_grad}, grad={bool(s.particle_x.grad) if s.particle_x.grad is not None else 'None'}")
print(f"  particle_v: req_grad={s.particle_v.requires_grad}, grad={bool(s.particle_v.grad) if s.particle_v.grad is not None else 'None'}")
print(f"  particle_stress: req_grad={s.particle_stress.requires_grad}, grad={bool(s.particle_stress.grad) if s.particle_stress.grad is not None else 'None'}")
print(f"  grid_v_out: req_grad={s.grid_v_out.requires_grad}, grad={bool(s.grid_v_out.grad) if s.grid_v_out.grad is not None else 'None'}")
print(f"  grid_v_in: req_grad={s.grid_v_in.requires_grad}, grad={bool(s.grid_v_in.grad) if s.grid_v_in.grad is not None else 'None'}")
print(f"  grid_m: req_grad={s.grid_m.requires_grad}, grad={bool(s.grid_m.grad) if s.grid_m.grad is not None else 'None'}")

# Create MoE params
mu_t = torch.ones(N, device=device) * 1e4
lam_t = torch.ones(N, device=device) * 1e4
weights_t = torch.ones(N, 4, device=device) * 0.25
fk_t = torch.ones(N, device=device) * 1e3
fdir_t = torch.zeros(N, 3, device=device)
fdir_t[:, 0] = 1.0

mu_t.requires_grad_(True)

moe_params = {
    'weights': torch2warp_float(weights_t, requires_grad=True),
    'mu': torch2warp_float(mu_t, requires_grad=True),
    'lam': torch2warp_float(lam_t, requires_grad=True),
    'fk': torch2warp_float(fk_t, requires_grad=True),
    'fdir': torch2warp_vec3(fdir_t, requires_grad=True),
    'active_mask': wp.array([1, 1, 1, 1], dtype=wp.int32, device=device)
}

# Single step with tape
tape = wp.Tape()
with tape:
    solver.p2g2p(0, 1e-4, device=device, moe_params=moe_params)

print(f"\n=== After p2g2p, tape has {len(tape.launches)} launches ===")

# Export and check
x_out = solver.export_particle_x_to_torch()
print(f"  x_out range: [{x_out.min().item():.4f}, {x_out.max().item():.4f}]")

# Check forward state
s_after = solver.mpm_state
grid_m_t = wp.to_torch(s_after.grid_m)
grid_v_out_t = wp.to_torch(s_after.grid_v_out)
stress_t = wp.to_torch(s_after.particle_stress)
selection_t = wp.to_torch(s_after.particle_selection, requires_grad=False)
v_t = wp.to_torch(s_after.particle_v)
print(f"  grid_m nonzero: {(grid_m_t > 0).sum().item()}, max: {grid_m_t.max().item():.6e}")
print(f"  grid_v_out norm: {torch.norm(grid_v_out_t).item():.6e}")
print(f"  stress norm: {torch.norm(stress_t.float()).item():.6e}")
print(f"  selection all zero: {(selection_t == 0).all().item()}")
print(f"  particle_v norm: {torch.norm(v_t).item():.6e}")

# Create seed gradient
seed = torch.ones_like(x_out) * 1e-3
seed_wp = torch2warp_vec3(seed)

# Set seed and backward
tape.backward(grads={s.particle_x: seed_wp})

print(f"\n=== After tape.backward ===")
print(f"  tape.gradients keys: {len(tape.gradients)}")

for name, arr in [('particle_x', s.particle_x), ('particle_v', s.particle_v), 
                   ('grid_v_out', s.grid_v_out), ('grid_v_in', s.grid_v_in),
                   ('stress', s.particle_stress), ('F_trial', s.particle_F_trial),
                   ('mu', moe_params['mu']), ('lam', moe_params['lam']),
                   ('weights', moe_params['weights'])]:
    in_grads = arr in tape.gradients
    if in_grads:
        g = tape.gradients[arr]
        g_norm = torch.norm(wp.to_torch(g).float()).item()
        print(f"  {name}: IN tape.gradients, norm={g_norm:.6e}")
    else:
        has_grad = arr.grad is not None
        if has_grad:
            g_norm = torch.norm(wp.to_torch(arr.grad).float()).item()
            print(f"  {name}: NOT in tape.gradients but .grad exists, norm={g_norm:.6e}")
        else:
            print(f"  {name}: NOT in tape.gradients, .grad=None, requires_grad={arr.requires_grad}")
