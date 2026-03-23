"""Minimal test: isolate which kernel backward produces NaN in Warp MPM tape."""
import torch
import warp as wp
import numpy as np
import sys
sys.path.insert(0, '.')

wp.init()
device = "cuda:0"

from phys_expert.model.diff_simulator.warp_solver.mpm_solver_warp import MPM_Simulator_WARP

N = 100

# Random positions in [0.3, 0.7] to be well inside the grid
pos = torch.rand(N, 3, device=device) * 0.4 + 0.3
vol = torch.ones(N, device=device) * 1e-6

solver = MPM_Simulator_WARP(n_particles=N, n_grid=32, grid_lim=1.0, device=device)
solver.load_initial_data_from_torch(pos, vol, n_grid=32, grid_lim=1.0, device=device)

# Set density so particle_mass is non-zero
solver.set_parameters_dict({"density": 1000.0, "E": 1e5, "nu": 0.3}, device=device)
solver.finalize_mu_lam(device=device)

# Boundary
solver.add_surface_collider(point=[0.5, 0.5, 0.5], normal=[0.0, 0.0, 1.0], surface="slip", friction=0.1)

s = solver.mpm_state
print(f"particle_x.requires_grad: {s.particle_x.requires_grad}")
print(f"particle_v.requires_grad: {s.particle_v.requires_grad}")
print(f"particle_mass nonzero: {(wp.to_torch(s.particle_mass) > 0).sum().item()}/{N}")
print(f"grid_m before p2g2p: {wp.to_torch(s.grid_m).sum().item():.6e}")

# Build MoE params (100% Neo-Hookean only)
mu_val = torch.full((N,), 5e4, device=device)
lam_val = torch.full((N,), 5e4, device=device)
fk_val = torch.full((N,), 1e3, device=device)
fdir_val = torch.zeros((N, 3), device=device)
fdir_val[:, 2] = 1.0
w_val = torch.zeros((N, 4), device=device)
w_val[:, 0] = 1.0

moe_params = {
    'mu': wp.from_torch(mu_val, dtype=wp.float32, requires_grad=True),
    'lam': wp.from_torch(lam_val, dtype=wp.float32, requires_grad=True),
    'fk': wp.from_torch(fk_val, dtype=wp.float32, requires_grad=True),
    'fdir': wp.from_torch(fdir_val, dtype=wp.vec3, requires_grad=True),
    'weights': wp.from_torch(w_val, requires_grad=True),
    'active_mask': wp.array([1, 0, 0, 0], dtype=wp.int32, device=device),
}

dt = 1e-4

# ========== TEST 1: Single p2g2p sub-step ==========
print("\n=== TEST 1: Single p2g2p sub-step ===")
tape = wp.Tape()
with tape:
    solver.p2g2p(0, dt, device=device, moe_params=moe_params, svd_clamp_max=2.0)

print(f"tape launches: {len(tape.launches)}")

# Check forward state
gm = wp.to_torch(s.grid_m)
gv = wp.to_torch(s.grid_v_out)
print(f"grid_m nonzero: {(gm > 0).sum().item()}, max: {gm.max().item():.4e}")
print(f"grid_v_out norm: {torch.norm(gv).item():.4e}")

# Seed gradient
seed = torch.randn(N, 3, device=device) * 1e-3
seed_wp = wp.from_torch(seed.contiguous(), dtype=wp.vec3)
print(f"seed norm: {torch.norm(seed).item():.4e}")

tape.backward(grads={s.particle_x: seed_wp})

checks = {
    'particle_x': s.particle_x,
    'particle_v': s.particle_v,
    'grid_v_out': s.grid_v_out,
    'grid_v_in': s.grid_v_in,
    'stress': s.particle_stress,
    'F_trial': s.particle_F_trial,
    'particle_F': s.particle_F,
    'mu': moe_params['mu'],
    'lam': moe_params['lam'],
}

for name, arr in checks.items():
    if arr in tape.gradients:
        g = tape.gradients[arr]
        gt = wp.to_torch(g).float()
        n = torch.norm(gt).item()
        has_nan = torch.isnan(gt).any().item()
        nan_count = torch.isnan(gt).sum().item()
        print(f"  {name:15s}: norm={n:.4e}, nan={has_nan} ({nan_count})")
    else:
        print(f"  {name:15s}: NOT in tape.gradients")

# ========== TEST 2: g2p kernel only ==========
print("\n=== TEST 2: g2p kernel only (no stress, no SVD) ===")
solver2 = MPM_Simulator_WARP(n_particles=N, n_grid=32, grid_lim=1.0, device=device)
solver2.load_initial_data_from_torch(pos, vol, n_grid=32, grid_lim=1.0, device=device)
solver2.set_parameters_dict({"density": 1000.0, "E": 1e5, "nu": 0.3}, device=device)
solver2.finalize_mu_lam(device=device)

s2 = solver2.mpm_state

# Put some non-zero grid velocities manually
grid_v_data = torch.randn(32, 32, 32, 3, device=device) * 0.001
wp.copy(s2.grid_v_out, wp.from_torch(grid_v_data.contiguous(), dtype=wp.vec3))

from phys_expert.model.diff_simulator.warp_solver.mpm_utils import g2p

tape2 = wp.Tape()
with tape2:
    wp.launch(g2p, dim=N, inputs=[s2, solver2.mpm_model, dt], device=device)

seed2 = torch.randn(N, 3, device=device) * 1e-3
seed2_wp = wp.from_torch(seed2.contiguous(), dtype=wp.vec3)

tape2.backward(grads={s2.particle_x: seed2_wp})

for name, arr in [('particle_x', s2.particle_x), ('particle_v', s2.particle_v), 
                   ('grid_v_out', s2.grid_v_out), ('particle_F', s2.particle_F)]:
    if arr in tape2.gradients:
        g = tape2.gradients[arr]
        gt = wp.to_torch(g).float()
        n = torch.norm(gt).item()
        has_nan = torch.isnan(gt).any().item()
        print(f"  {name:15s}: norm={n:.4e}, nan={has_nan}")
    else:
        print(f"  {name:15s}: NOT in tape2.gradients")

print("\nDone.")
