import torch
import os
import sys
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phys_expert.model.router_network import PhysicsRouterNetwork

def test_router_forward():
    print("Loading Config...")
    cfg_path = os.path.join(os.path.dirname(__file__), "phys_expert_jelly.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    # Instantiate Router
    print("Instantiating PhysicsRouterNetwork...")
    router_cfg = cfg.router
    
    # Ensure nested configs are accessible as DictConfig expects
    # OmegaConf loaded object is already DictConfig
    
    router = PhysicsRouterNetwork(router_cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router.to(device)
    print(f"Model on {device}")
    
    # Create Dummy Inputs
    B = 2
    N_gaussians = 1000
    N_tracks = 500 
    T = 16
    
    # Gaussians: [B, N, 6]
    # 0-3: XYZ, 3-6: Color
    gaussians = torch.randn(B, N_gaussians, 6).to(device)
    
    # Tracks: [B, T, N, 3]
    tracks = torch.randn(B, T, N_tracks, 3).to(device)
    
    print(f"Input Shapes: Gaussians {gaussians.shape}, Tracks {tracks.shape}")
    
    # Forward
    weights, params = router(gaussians, tracks)
    
    print("Forward Pass Successful!")
    print(f"Weights Shape: {weights.shape}") # Should be [B, K, 4]
    print(f"Params Shape: {params.shape}")   # Should be [B, K, 8]
    
    # Verify K
    K = router_cfg.static.n_patches
    assert weights.shape == (B, K, 4)
    assert params.shape == (B, K, 8)
    
    print("Test Passed!")

if __name__ == "__main__":
    test_router_forward()
