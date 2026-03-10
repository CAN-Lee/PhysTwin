import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points
from .point_transformer import HierarchicalPointTransformer, SinusoidalPositionalEmbedding
from .temporal_transformer import SpatiotemporalTransformer

class FusionModule(nn.Module):
    """Cross-Attention Fusion Module"""
    def __init__(self, d_model):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Spatial Positional Encoding for Query (Dynamic Features)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model)
        
        # Heads
        self.weight_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 4), # 4 Experts
            nn.Softmax(dim=-1)
        )
        self.param_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8), # mu, lam, fiber_k, fiber_dir(3), friction, restitution
            nn.Softplus() # Enforce positive params
        )

    def forward(self, static_feat, dynamic_feat, patch_centers):
        """
        static_feat: [B, K, D] (Key/Value) - Already has spatial PE
        dynamic_feat: [B, K, D] (Query) - Needs spatial PE
        patch_centers: [B, K, 3] - Coordinates for PE
        """
        # Add Spatial Position Encoding to Dynamic Features (Query)
        # This allows the query to know "where" it is looking from
        query_pe = self.pos_emb(patch_centers)
        query = dynamic_feat + query_pe
        
        # dynamic (Query) attends to static (Key/Value)
        # Note: Static features already contain their own spatial PE from PointTransformer
        attn_out, _ = self.cross_attn(query, static_feat, static_feat)
        
        fused = self.norm(dynamic_feat + attn_out)
        fused = fused + self.ffn(fused)
        
        weights = self.weight_head(fused)
        params = self.param_head(fused)
        
        # --- Stability Guard: Clamp physical parameters ---
        # Prevent extreme values that cause numerical explosion (NaN)
        # mu, lam, fiber_k: [0, 1e5] (Lowered from 1e6 for stability)
        # friction, restitution: [0, 1]
        params = torch.clamp(params, max=1e5) 
        
        return weights, params

class PhysicsRouterNetwork(nn.Module):
    """
    Main Router Network: Two-Stream Architecture
    """
    def __init__(self, cfg):
        super().__init__()
        
        d_model = getattr(cfg, 'd_model', 128)
        self.n_patches = getattr(cfg.static, 'n_patches', 64) # Global patch count
        
        # Handle Static Config
        if not hasattr(cfg.static, 'd_model'):
            try: cfg.static.d_model = d_model
            except: pass
                
        # Handle Dynamic Config
        if not hasattr(cfg.dynamic, 'd_model'):
            try: cfg.dynamic.d_model = d_model
            except: pass
        
        # Sync n_patches
        if not hasattr(cfg.dynamic, 'n_patches'):
             try: cfg.dynamic.n_patches = self.n_patches
             except: pass

        self.static_stream = HierarchicalPointTransformer(cfg.static)
        self.dynamic_stream = SpatiotemporalTransformer(cfg.dynamic)
        self.fusion = FusionModule(d_model)

    def forward(self, gaussians, tracks):
        """
        Args:
            gaussians: [B, N, 14] Initial Gaussian attributes (Static)
            tracks: [B, T, N, 3] Point tracks (Dynamic)
        Returns:
            weights: [B, K, 4] Patch-wise expert weights
            params: [B, K, 8] Patch-wise material params
        """
        # 1. Unified Patch Sampling (Alignment)
        # Sample patch centers from the first frame geometry (Gaussians XYZ)
        xyz_static = gaussians[..., :3] # [B, N_g, 3]
        
        # We use FPS on static geometry to define patches
        patch_centers, _ = sample_farthest_points(xyz_static, K=self.n_patches) # [B, K, 3]
        
        # 2. Static Stream
        # Pass patch_centers to enforce alignment
        static_feat = self.static_stream(gaussians, patch_centers=patch_centers) # [B, K, D]
        
        # 3. Dynamic Stream
        # Pass patch_centers to gather corresponding tracks
        # Note: tracks are dense [B, T, N_t, 3]. 
        # The centers are in the same coordinate space as tracks at t=0.
        dynamic_feat = self.dynamic_stream(tracks, patch_centers=patch_centers)  # [B, K, D]
        
        # 4. Fusion
        # Pass patch_centers for spatial PE on query
        weights, params = self.fusion(static_feat, dynamic_feat, patch_centers)
        
        return weights, params
