
import torch
import os
import sys
import argparse
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Adjust path to import phys_expert
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phys_expert.engine.simulator_mpm import MPMSimulator

# Reuse util functions from mpm_pytorch examples if possible, or reimplement simply
# Here reimplementing simple cube generator and visualizer to be self-contained

def get_cube(center, size, num, device):
    start = torch.tensor(center) - torch.tensor(size) / 2
    end = torch.tensor(center) + torch.tensor(size) / 2
    x = torch.linspace(start[0], end[0], num)
    y = torch.linspace(start[1], end[1], num)
    z = torch.linspace(start[2], end[2], num)
    cube = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).view(-1, 3)
    return cube.to(device)

def visualize_frames(frames, export_path, fps=30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Static limits based on jelly example
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    
    scat = ax.scatter([], [], [], s=5, c='blue')
    
    def update(frame):
        data = frames[frame]
        scat._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
        ax.set_title(f'Frame {frame}')
        return scat
        
    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(export_path, writer='pillow', fps=fps)
    plt.close()

def test_jelly_drop():
    print("Testing Jelly Drop with MPMSimulator (PhysTwin Wrapper)...")
    
    # 1. Config based on jelly.yaml
    yaml_path = os.path.join(os.path.dirname(__file__), "phys_expert_jelly.yaml")
    if os.path.exists(yaml_path):
        cfg_file = OmegaConf.load(yaml_path)
        # Merge basic settings, but here we construct a simple cfg manually for testing
        # To use the yaml fully, we'd do: cfg = cfg_file.mpm
        # Let's just override specific values for this test to show integration
        dt = cfg_file.mpm.dt
        grid_res = cfg_file.mpm.grid_res
        max_frames = cfg_file.mpm.max_frames
        steps_per_frame = cfg_file.mpm.steps_per_frame
    else:
        dt = 1e-4
        grid_res = 64
        max_frames = 60
        steps_per_frame = 20

    cfg = OmegaConf.create({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dt": dt,
        "grid_res": grid_res,
        "box_size": 1.0,
        "n_particles": 500 # Placeholder, will be updated by reset
    })
    
    device = torch.device(cfg.device)
    
    # 2. Particles
    # Center 0.5, 0.5, 0.5 (drop from mid-air)
    particles = get_cube([0.5, 0.5, 0.4], [0.2, 0.2, 0.2], num=15, device=device)
    cfg.n_particles = particles.shape[0]
    
    # 3. Initialize Simulator
    simulator = MPMSimulator(cfg)
    simulator.reset(particles)
    
    # Set Initial Velocity (downward push)
    simulator.mpm_solver.v = torch.tensor([0.0, 0.0, -0.5], device=device).repeat(cfg.n_particles, 1)
    
    # 4. Construct Expert Parameters
    # Target: Corotated Elasticity with E=2e6, nu=0.4
    # Conversion:
    # mu = E / (2*(1+nu)) = 2e6 / 2.8 = 7.14e5
    # lam = E*nu / ((1+nu)*(1-2nu)) = 2e6*0.4 / (1.4*0.2) = 8e5 / 0.28 = 2.86e6
    
    N = cfg.n_particles
    mu_val = 7.14e5
    lam_val = 2.86e6
    
    # Activate Expert B (Corotated) -> Weights [0, 1, 0, 0]
    weights = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).repeat(N, 1)
    
    params = {
        "mu": torch.full((N,), mu_val, device=device),
        "lam": torch.full((N,), lam_val, device=device),
        "fiber_k": torch.zeros(N, device=device),
        "fiber_dir": torch.zeros((N, 3), device=device)
    }
    
    # 5. Simulation Loop
    frames = []
    
    from tqdm import tqdm
    for f in tqdm(range(max_frames)):
        frames.append(simulator.x.cpu().numpy())
        for s in range(steps_per_frame):
            simulator.step(weights, params)
            
    # 6. Export
    os.makedirs("output", exist_ok=True)
    visualize_frames(frames, "output/test_phys_expert_jelly.gif")
    print("Done! Saved to output/test_phys_expert_jelly.gif")

if __name__ == "__main__":
    test_jelly_drop()
