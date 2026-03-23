import torch
import warp as wp
import numpy as np
import os
import sys
import yaml
from easydict import EasyDict as ConfigDict

# Add required paths
sys.path.append(os.getcwd())

from phys_expert.model.experts.mixture_model import MixtureElasticity
from phys_expert.model.diff_simulator.warp_solver.moe_utils import compute_moe_stress_kernel
from phys_expert.model.diff_simulator.warp_solver.warp_utils import torch2warp_float, torch2warp_vec3, torch2warp_mat33

def test_moe_stress_consistency():
    wp.init()
    device = "cuda:0"
    n_particles = 10
    
    # 1. Setup Parameters
    active_experts = ['nh', 'co', 'st', 'fi']
    expert_weights = torch.softmax(torch.randn(n_particles, 4), dim=-1).cuda()
    mu = torch.exp(torch.randn(n_particles)).cuda() * 1e4
    lam = torch.exp(torch.randn(n_particles)).cuda() * 1e4
    fk = torch.exp(torch.randn(n_particles)).cuda() * 1e4
    fdir = torch.nn.functional.normalize(torch.randn(n_particles, 3), dim=-1).cuda()
    
    F = torch.eye(3).repeat(n_particles, 1, 1).cuda() + torch.randn(n_particles, 3, 3).cuda() * 0.1
    
    expert_params = {
        'weights': expert_weights,
        'mu': mu,
        'lam': lam,
        'fiber_k': fk,
        'fiber_dir': fdir
    }
    
    # 2. Compute Stress in PyTorch
    print("Computing stress in PyTorch...")
    model_torch = MixtureElasticity(active_experts=active_experts).cuda()
    model_torch.current_params = expert_params
    P_torch = model_torch(F)
    # Convert P to Kirchhoff Stress tau = P F^T for comparison with Warp
    tau_torch = torch.matmul(P_torch, F.transpose(-2, -1))
    
    # 3. Compute Stress in Warp
    print("Computing stress in Warp...")
    particle_F_wp = torch2warp_mat33(F)
    particle_weights_wp = torch2warp_float(expert_weights)
    particle_mu_wp = torch2warp_float(mu)
    particle_lam_wp = torch2warp_float(lam)
    particle_fk_wp = torch2warp_float(fk)
    particle_fdir_wp = torch2warp_vec3(fdir)
    particle_stress_wp = wp.zeros(n_particles, dtype=wp.mat33, device=device)
    active_mask_wp = wp.array([1, 1, 1, 1], dtype=int, device=device)
    
    wp.launch(
        kernel=compute_moe_stress_kernel,
        dim=n_particles,
        inputs=[
            particle_F_wp,
            particle_weights_wp,
            particle_mu_wp,
            particle_lam_wp,
            particle_fk_wp,
            particle_fdir_wp,
            particle_stress_wp,
            active_mask_wp
        ],
        device=device
    )
    
    tau_warp = wp.to_torch(particle_stress_wp).reshape(n_particles, 3, 3)
    
    # 4. Compare
    diff = (tau_torch - tau_warp).abs().mean().item()
    print(f"\nMean Absolute Difference: {diff:.2e}")
    
    if diff < 1e-4:
        print("SUCCESS: Stress consistency verified!")
    else:
        print("FAILURE: Stress difference too large.")
        # Print first particle comparison
        print("\nParticle 0 Torch tau:")
        print(tau_torch[0])
        print("\nParticle 0 Warp tau:")
        print(tau_warp[0])

if __name__ == "__main__":
    test_moe_stress_consistency()
