import torch
import yaml
from easydict import EasyDict as ConfigDict
import os
import sys
import time

# Add required paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "phys_expert/model/diff_simulator/warp_solver"))

from phys_expert.engine.simulator_mpm import MPMSimulator
from phys_expert.model.diff_simulator.warp_solver.simulator_warp import WarpMPMSimulator

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return ConfigDict(cfg)

def compare_simulators():
    cfg = load_config('configs/softbody.yaml')
    cfg.mpm.device = 'cuda'
    
    # Ensure numerical values are correctly typed
    cfg.mpm.dt = float(cfg.mpm.dt)
    cfg.mpm.grid_res = int(cfg.mpm.grid_res)
    
    print(f"DEBUG: cfg.mpm.dt type: {type(cfg.mpm.dt)}, value: {cfg.mpm.dt}")
    
    # Override n_particles for test speed
    cfg.mpm.n_particles = 1000
    
    # 1. Initialize Particles
    init_particles = torch.randn((cfg.mpm.n_particles, 3), device='cuda') * 0.05
    init_particles[:, 2] += 0.2 # Float above ground
    
    # 2. Setup Current PyTorch Simulator
    print("Initializing PyTorch MPMSimulator...")
    torch_sim = MPMSimulator(cfg.mpm)
    torch_sim.reset(init_particles)
    
    # 3. Setup New Warp Simulator
    print("Initializing Warp WarpMPMSimulator...")
    warp_sim = WarpMPMSimulator(cfg.mpm)
    warp_sim.reset(init_particles)
    
    # 4. Prepare expert params
    expert_weights = torch.ones((cfg.mpm.n_particles, 4), device='cuda') * 0.25
    expert_params = {
        'mu': torch.ones(cfg.mpm.n_particles, device='cuda') * 1e4,
        'lam': torch.ones(cfg.mpm.n_particles, device='cuda') * 1e4,
        'fiber_k': torch.zeros(cfg.mpm.n_particles, device='cuda'),
        'fiber_dir': torch.tensor([1.0, 0.0, 0.0], device='cuda').repeat(cfg.mpm.n_particles, 1),
        'yield_stress': None,
        'plastic_viscosity': None
    }
    
    # 5. Run Steps and Profile
    n_steps = 10
    
    print(f"\nRunning {n_steps} steps with PyTorch Simulator...")
    start_time = time.time()
    for i in range(n_steps):
        torch_x = torch_sim.step(expert_weights, expert_params)
    torch_duration = time.time() - start_time
    print(f"PyTorch duration: {torch_duration:.4f}s ({torch_duration/n_steps:.4f}s/step)")
    
    print(f"\nRunning {n_steps} steps with Warp Simulator...")
    start_time = time.time()
    for i in range(n_steps):
        warp_x = warp_sim.step(expert_weights, expert_params)
    warp_duration = time.time() - start_time
    print(f"Warp duration: {warp_duration:.4f}s ({warp_duration/n_steps:.4f}s/step)")
    
    # 6. Check consistency
    print(f"PyTorch x mean: {torch_x.mean().item()}, nan count: {torch.isnan(torch_x).sum().item()}")
    print(f"Warp x mean: {warp_x.mean().item()}, nan count: {torch.isnan(warp_x).sum().item()}")
    
    diff = (torch_x - warp_x).abs().mean().item()
    print(f"\nFinal position mean difference: {diff:.6f}")
    
    if warp_duration < torch_duration:
        print(f"Speedup: {torch_duration/warp_duration:.2f}x")
    else:
        print("Warp was not faster in this small-scale test (likely due to overhead or small N).")

if __name__ == "__main__":
    compare_simulators()
