"""Test tape backward with increasing number of sub-steps to find where NaN appears."""
import torch
import warp as wp
import sys
sys.path.insert(0, '.')
wp.init()
device = "cuda:0"

from phys_expert.model.diff_simulator.warp_solver.mpm_solver_warp import MPM_Simulator_WARP

N = 100
pos = torch.rand(N, 3, device=device) * 0.4 + 0.3
vol = torch.ones(N, device=device) * 1e-6
dt = 1e-4

for n_steps in [1, 5, 10, 50, 100, 333]:
    solver = MPM_Simulator_WARP(n_particles=N, n_grid=32, grid_lim=1.0, device=device)
    solver.load_initial_data_from_torch(pos, vol, n_grid=32, grid_lim=1.0, device=device)
    solver.set_parameters_dict({"density": 1000.0, "E": 1e5, "nu": 0.3}, device=device)
    solver.finalize_mu_lam(device=device)
    solver.add_surface_collider(point=[0.5, 0.5, 0.5], normal=[0.0, 0.0, 1.0], surface="slip", friction=0.1)

    s = solver.mpm_state
    
    mu_val = torch.full((N,), 5e4, device=device)
    lam_val = torch.full((N,), 5e4, device=device)
    fk_val = torch.full((N,), 1e3, device=device)
    fdir_val = torch.zeros((N, 3), device=device); fdir_val[:, 2] = 1.0
    w_val = torch.zeros((N, 4), device=device); w_val[:, 0] = 1.0

    moe_params = {
        'mu': wp.from_torch(mu_val, dtype=wp.float32, requires_grad=True),
        'lam': wp.from_torch(lam_val, dtype=wp.float32, requires_grad=True),
        'fk': wp.from_torch(fk_val, dtype=wp.float32, requires_grad=True),
        'fdir': wp.from_torch(fdir_val, dtype=wp.vec3, requires_grad=True),
        'weights': wp.from_torch(w_val, requires_grad=True),
        'active_mask': wp.array([1, 0, 0, 0], dtype=wp.int32, device=device),
    }

    tape = wp.Tape()
    with tape:
        for _ in range(n_steps):
            solver.p2g2p(0, dt, device=device, moe_params=moe_params, svd_clamp_max=2.0)

    seed = torch.randn(N, 3, device=device) * 1e-3
    seed_wp = wp.from_torch(seed.contiguous(), dtype=wp.vec3)
    tape.backward(grads={s.particle_x: seed_wp})

    results = {}
    for name, arr in [('x', s.particle_x), ('v', s.particle_v), ('stress', s.particle_stress),
                       ('mu', moe_params['mu']), ('lam', moe_params['lam'])]:
        if arr in tape.gradients:
            gt = wp.to_torch(tape.gradients[arr]).float()
            n = torch.norm(gt).item()
            has_nan = torch.isnan(gt).any().item()
            results[name] = f"{n:.2e}{'(NaN!)' if has_nan else ''}"
        else:
            results[name] = "N/A"

    parts = [f"{k}={v}" for k, v in results.items()]
    print(f"steps={n_steps:4d} | {' | '.join(parts)}")

print("\nDone.")
