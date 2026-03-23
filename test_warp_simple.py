import torch
import warp as wp
import sys
import os

# Add the local directory to sys.path to allow relative-like imports in the copied files
sys.path.append(os.path.join(os.getcwd(), "phys_expert/model/diff_simulator/warp_solver"))

from phys_expert.model.diff_simulator.warp_solver.mpm_solver_warp import MPM_Simulator_WARP

def test_warp():
    wp.init()
    device = "cuda:0"
    
    # 1. Create dummy particles
    n_particles = 100
    pos = torch.rand((n_particles, 3), device="cuda") * 0.1 + 0.5 # Center [0.5, 0.5, 0.5]
    vol = torch.ones(n_particles, device="cuda") * (0.01**3)
    
    # 2. Init Warp Solver
    print("Initializing Warp Solver...")
    mpm_solver = MPM_Simulator_WARP(n_particles)
    mpm_solver.load_initial_data_from_torch(
        pos, 
        vol, 
        n_grid=32, 
        grid_lim=1.0, 
        device=device
    )
    
    # 3. Set parameters
    material_params = {
        "material": "elastic",
        "E": 1e5,
        "nu": 0.3,
        "density": 1000.0,
        "g": [0.0, 0.0, -9.8]
    }
    mpm_solver.set_parameters_dict(material_params, device=device)
    mpm_solver.finalize_mu_lam(device=device)
    
    # 4. Run one step
    dt = 1e-4
    print("Running one step...")
    mpm_solver.p2g2p(0, dt, device=device)
    
    # 5. Export results
    new_pos = mpm_solver.export_particle_x_to_torch()
    print(f"Step completed. New position mean: {new_pos.mean().item()}")
    
    if not torch.allclose(pos, new_pos):
        print("Success: Particles moved!")
    else:
        print("Warning: Particles did not move (maybe dt is too small or force is zero).")

if __name__ == "__main__":
    test_warp()
