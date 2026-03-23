import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
import os

# [NEW] Import local PGND components
from .pgnd_components.encoder import PointNetEncoder
from .pgnd_components.decoder import CondNeRFModel
from .pgnd_components.utils import get_bspline_weights, positional_encoding

class ResidualPGND(nn.Module):
    """
    Residual Dynamics Model (Scheme A): Feedback-style correction.
    Inputs are the results from the MPM solver + history.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Hyperparameters
        self.feature_dim = getattr(cfg, 'feature_dim', 64)
        self.n_history = getattr(cfg, 'n_history', 2)
        self.num_grids_list = getattr(cfg, 'grid_res_list', [64, 64, 64])
        self.dx = 1.0 / self.num_grids_list[0]
        
        self.pe_num_func = getattr(cfg, 'pe_num_func', 6)
        self.pe_dim = 3 + self.pe_num_func * 6
        
        self.input_scale = getattr(cfg, 'input_scale', 1.0)
        self.output_scale = getattr(cfg, 'output_scale', 0.05) 
        
        # 1. Point Encoder (Lagrangian)
        # Input channel: 9 (pos_mpm, vel_mpm, disp_phys) + 6 * n_history
        input_channels = 9 + 6 * self.n_history
        self.encoder = PointNetEncoder(
            global_feat=False, 
            feature_dim=self.feature_dim,
            channel=input_channels
        )
        
        # 2. Neural Field Decoder (Eulerian)
        # Output channel: 4 (3 for delta_v, 1 for damping gamma)
        self.decoder = CondNeRFModel(
            xyz_dim=self.pe_dim,
            condition_dim=self.feature_dim,
            out_channel=4,
            num_layers=getattr(cfg, 'num_layers', 4),
            hidden_size=getattr(cfg, 'hidden_size', 128),
            skip_connect_every=4
        )

    def forward(self, x_mpm: Tensor, v_mpm: Tensor, x_start: Tensor, x_his: Tensor, v_his: Tensor, mode: str = "residual") -> Tensor:
        """
        Predict correction based on MPM solver results.
        Args:
            x_mpm: positions after MPM solver [B, N, 3]
            v_mpm: velocities after MPM solver [B, N, 3]
            x_start: positions at the start of this frame [B, N, 3]
            x_his: history positions [B, N, H, 3]
            v_his: history velocities [B, N, H, 3]
            mode: "residual" (returns delta_v) or "damping" (returns gamma) or "both"
        """
        bsz, num_particles, _ = x_mpm.shape
        device = x_mpm.device
        
        # [NEW] Dynamic Centering: Implementation of translation invariance
        # Calculate centroid of current frame
        x_center = x_mpm.mean(dim=1, keepdim=True) # [B, 1, 3]
        
        # Center all positional inputs
        x_mpm_norm = x_mpm - x_center
        x_start_norm = x_start - x_center
        x_his_norm = x_his - x_center.unsqueeze(2) # x_his is [B, N, H, 3]
        
        # 1. Calculate Differential Features
        # Note: Displacement and Velocity are already translation-invariant vectors
        disp_phys = (x_mpm - x_start) * self.input_scale
        v_mpm_scaled = v_mpm * self.input_scale
        
        x_his_flat = x_his_norm.reshape(bsz, num_particles, -1)
        v_his_flat = v_his.reshape(bsz, num_particles, -1)
        
        # 2. Assemble Input Features
        # Using normalized positions for x_mpm and x_his
        feat_in = torch.cat([x_mpm_norm, v_mpm_scaled, disp_phys, x_his_flat, v_his_flat], dim=-1)
        feat_in = feat_in.permute(0, 2, 1) # [B, D, N] for PointNet
        
        # 3. B-Spline Interpolation Setup (Aligned with MPM)
        # Weights and node indices depend on absolute grid position, so we use raw x_mpm
        weights, node_idxs = get_bspline_weights(x_mpm, self.dx, self.num_grids_list)
        
        # 4. Lagrangian Feature Extraction
        feat, _ = self.encoder(feat_in)
        feat = feat.permute(0, 2, 1) # [B, N, feature_dim]
        
        # 5. Eulerian Field Prediction at Neighboring Nodes
        # IMPORTANT: The coordinate input to the Neural Field must ALSO be centered
        node_pos = node_idxs.float() * self.dx 
        node_pos_norm = node_pos - x_center.unsqueeze(2) # Center node positions relative to object
        
        node_pos_flat = node_pos_norm.reshape(bsz, -1, 3)
        node_feat_flat = feat.unsqueeze(2).expand(-1, -1, 27, -1).reshape(bsz, -1, self.feature_dim)
        
        node_pe_flat = positional_encoding(node_pos_flat, self.pe_num_func)
        
        raw_output = self.decoder(node_pe_flat, node_feat_flat)
        
        # Branch 1: delta_v [B, N_pts * 27, 3]
        delta_v_node_flat = torch.tanh(raw_output[..., :3])
        delta_v_node = delta_v_node_flat.reshape(bsz, num_particles, 27, 3)
        delta_v_p = torch.sum(weights.unsqueeze(-1) * delta_v_node, dim=2) * self.output_scale
        
        # Branch 2: damping gamma [B, N_pts * 27, 1]
        # Map to [0, 0.1] for safety
        gamma_node_flat = torch.sigmoid(raw_output[..., 3:]) * 0.1
        gamma_node = gamma_node_flat.reshape(bsz, num_particles, 27, 1)
        gamma_p = torch.sum(weights.unsqueeze(-1) * gamma_node, dim=2)
        
        if mode == "residual":
            return delta_v_p
        elif mode == "damping":
            return gamma_p
        elif mode == "both":
            return delta_v_p, gamma_p
        else:
            raise ValueError(f"Unknown mode: {mode}")
