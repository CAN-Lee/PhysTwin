import torch
import torch.nn as nn
import torch.optim as optim
import warp as wp
import os
import sys
import yaml
from easydict import EasyDict as ConfigDict

# Add required paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "phys_expert/model/diff_simulator/warp_solver"))

from phys_expert.model.diff_simulator.warp_solver.simulator_warp import WarpMPMSimulator
from phys_expert.model.residual_pgnd import ResidualPGND

def test_warp_e2e():
    wp.init()
    device = "cuda:0"
    
    # 1. Configuration
    with open('configs/softbody.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = ConfigDict(cfg)
    cfg.mpm.device = 'cuda'
    cfg.mpm.n_particles = 1000
    cfg.mpm.grid_res = 32
    
    cfg.mpm.dt = float(cfg.mpm.dt)
    cfg.mpm.grid_res = int(cfg.mpm.grid_res)
    
    # 2. Initialize Models
    print("Initializing Warp Simulator...")
    simulator = WarpMPMSimulator(cfg.mpm).to("cuda")
    
    print("Initializing ResidualPGND...")
    # Update residual config to match n_particles if needed
    residual_net = ResidualPGND(cfg.residual).to("cuda")
    
    # 3. Setup Initial State
    init_particles = torch.randn((cfg.mpm.n_particles, 3), device='cuda') * 0.05
    init_particles[:, 2] += 0.2
    
    # Add dummy controller
    controller_pos = torch.tensor([[0.0, 0.0, 0.3]], device='cuda')
    simulator.reset(init_particles, controller_pos=controller_pos)
    
    # Dummy history for PGND
    H = cfg.residual.n_history
    x_history = [init_particles.clone() for _ in range(H)]
    v_history = [torch.zeros_like(init_particles) for _ in range(H)]
    x_start_frame = init_particles.unsqueeze(0)
    
    # 4. End-to-End Forward Pass with Gradient Recording
    print("\nRunning E2E Forward Pass...")
    tape = wp.Tape()
    simulator.tape = tape
    
    # Simulate one step with residual injection
    curr_x_mpm = (simulator.x - simulator.shift).unsqueeze(0)
    curr_v_mpm = simulator.v.unsqueeze(0)
    x_his_tensor = torch.stack(x_history, dim=1).unsqueeze(0)
    v_his_tensor = torch.stack(v_history, dim=1).unsqueeze(0)
    
    # Predict residual
    delta_v = residual_net(curr_x_mpm, curr_v_mpm, x_start_frame, x_his_tensor, v_his_tensor, mode="residual").squeeze(0)
    delta_v.retain_grad()
    
    print(f"delta_v requires_grad: {delta_v.requires_grad}")
    
    # Step simulator with residual
    expert_weights = torch.ones((cfg.mpm.n_particles, 4), device='cuda') * 0.25
    expert_weights.requires_grad = True
    expert_params = {
        'mu': torch.ones(cfg.mpm.n_particles, device='cuda') * 1e4,
        'lam': torch.ones(cfg.mpm.n_particles, device='cuda') * 1e4,
        'fiber_k': torch.ones(cfg.mpm.n_particles, device='cuda') * 1e4,
        'fiber_dir': torch.tensor([1.0, 0.0, 0.0], device='cuda').repeat(cfg.mpm.n_particles, 1),
        'yield_stress': torch.zeros(cfg.mpm.n_particles, device='cuda'), # Not used in MoE stress directly
        'plastic_viscosity': torch.zeros(cfg.mpm.n_particles, device='cuda')
    }
    
    # Enable gradients for expert_params
    expert_params['mu'].requires_grad = True
    expert_params['lam'].requires_grad = True
    expert_params['fiber_k'].requires_grad = True
    expert_params['fiber_dir'].requires_grad = True
    
    x_next = simulator.step(expert_weights, expert_params, residual_v=delta_v)
    x_next.retain_grad()
    
    print(f"Simulator v requires_grad: {simulator.v.requires_grad}")
    print(f"Warp particle_v requires_grad: {simulator.solver.mpm_state.particle_v.requires_grad}")
    
    # 5. Compute Dummy Loss
    target_pos = init_particles + torch.tensor([0.0, 0.0, 0.01], device='cuda') # Small target movement
    loss = torch.mean((x_next - target_pos)**2)
    print(f"Initial Loss: {loss.item():.6f}")
    
    # 6. Backward Pass
    print("\nRunning Backward Pass...")
    # Step A: Torch backward to compute grad of simulator.x
    loss.backward()
    
    print(f"x_next grad mean: {x_next.grad.abs().mean().item():.2e}")
    
    # Step B: Warp Tape backward to propagate grad to simulator.v (which is a Torch tensor)
    # We pass the gradients from simulator.x.grad into the tape
    from phys_expert.model.diff_simulator.warp_solver.warp_utils import torch2warp_vec3
    
    # Create gradient arrays for Tape
    grads_dict = {
        simulator.solver.mpm_state.particle_x: torch2warp_vec3(simulator.x.grad)
    }
    
    tape.backward(grads=grads_dict)
    
    # Step C: Check which gradients are available
    print("\nAvailable gradients in Tape:")
    found_v = False
    v_ptr = simulator.solver.mpm_state.particle_v.ptr
    x_ptr = simulator.solver.mpm_state.particle_x.ptr
    print(f"  Target particle_v ptr: {v_ptr}")
    print(f"  Target particle_x ptr: {x_ptr}")
    
    for array, grad in tape.gradients.items():
        if hasattr(array, "ptr") and hasattr(array, "shape") and hasattr(array, "dtype"):
            if array.shape == (simulator.n_particles,) and array.dtype == wp.vec3:
                 print(f"  Found array in Tape: ptr={array.ptr}, shape={array.shape}")
                 if array.ptr == v_ptr:
                      found_v = True
                      print("  -> This matches particle_v!")
                      grad_v_from_tape = wp.to_torch(grad)
                      delta_v.backward(grad_v_from_tape)
                 elif array.ptr == x_ptr:
                      print("  -> This matches particle_x!")

    if not found_v:
        print("  -> particle_v gradient NOT found in Tape!")
    
    # 7. Verify Gradient Flow to NN
    has_grads = False
    for name, p in residual_net.named_parameters():
        if p.grad is not None:
            has_grads = True
            print(f"Param {name} grad mean: {p.grad.abs().mean().item():.2e}")
            break
    
    if has_grads:
        print("\nSUCCESS: End-to-end gradient flow verified!")
    else:
        print("\nFAILURE: Gradient flow blocked.")

    # 8. Verify Gradient Flow to Expert Parameters
    print("\nVerifying gradients for expert parameters...")
    if expert_weights.grad is not None:
        print(f"expert_weights grad mean: {expert_weights.grad.abs().mean().item():.2e}")
    else:
        print("expert_weights grad is None")
        
    for key in ['mu', 'lam', 'fiber_k', 'fiber_dir']:
        if expert_params[key].grad is not None:
            print(f"expert_params[{key}] grad mean: {expert_params[key].grad.abs().mean().item():.2e}")
        else:
            print(f"expert_params[{key}] grad is None")

if __name__ == "__main__":
    test_warp_e2e()
