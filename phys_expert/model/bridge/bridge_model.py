from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_bank import aggregate_anchor_positions


@dataclass
class BridgeConfig:
    gaussian_anchor_k: int = 8
    anchor_graph_k: int = 8
    max_pos_residual: float = 0.005
    max_opacity_delta: float = 0.5
    residual_hidden_size: int = 128
    residual_layers: int = 3


def _normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(q, dim=-1, eps=1e-8)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(q1)
    out[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    out[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    out[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    out[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
    return out


def _rotation_matrix_to_quaternion(rot: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(rot.shape[0], 4, device=rot.device, dtype=rot.dtype)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    positive = trace > 0.0

    if positive.any():
        t = torch.sqrt(trace[positive] + 1.0) * 2.0
        q[positive, 0] = 0.25 * t
        q[positive, 1] = (rot[positive, 2, 1] - rot[positive, 1, 2]) / t
        q[positive, 2] = (rot[positive, 0, 2] - rot[positive, 2, 0]) / t
        q[positive, 3] = (rot[positive, 1, 0] - rot[positive, 0, 1]) / t

    mask = ~positive
    if mask.any():
        rot_m = rot[mask]
        diag = torch.stack([rot_m[:, 0, 0], rot_m[:, 1, 1], rot_m[:, 2, 2]], dim=1)
        max_idx = torch.argmax(diag, dim=1)
        for axis in range(3):
            axis_mask = max_idx == axis
            if not axis_mask.any():
                continue
            sub = rot_m[axis_mask]
            if axis == 0:
                t = torch.sqrt(1.0 + sub[:, 0, 0] - sub[:, 1, 1] - sub[:, 2, 2]) * 2.0
                q_sel = torch.stack(
                    [
                        (sub[:, 2, 1] - sub[:, 1, 2]) / t,
                        0.25 * t,
                        (sub[:, 0, 1] + sub[:, 1, 0]) / t,
                        (sub[:, 0, 2] + sub[:, 2, 0]) / t,
                    ],
                    dim=1,
                )
            elif axis == 1:
                t = torch.sqrt(1.0 + sub[:, 1, 1] - sub[:, 0, 0] - sub[:, 2, 2]) * 2.0
                q_sel = torch.stack(
                    [
                        (sub[:, 0, 2] - sub[:, 2, 0]) / t,
                        (sub[:, 0, 1] + sub[:, 1, 0]) / t,
                        0.25 * t,
                        (sub[:, 1, 2] + sub[:, 2, 1]) / t,
                    ],
                    dim=1,
                )
            else:
                t = torch.sqrt(1.0 + sub[:, 2, 2] - sub[:, 0, 0] - sub[:, 1, 1]) * 2.0
                q_sel = torch.stack(
                    [
                        (sub[:, 1, 0] - sub[:, 0, 1]) / t,
                        (sub[:, 0, 2] + sub[:, 2, 0]) / t,
                        (sub[:, 1, 2] + sub[:, 2, 1]) / t,
                        0.25 * t,
                    ],
                    dim=1,
                )
            q[mask.nonzero(as_tuple=False).squeeze(1)[axis_mask]] = q_sel

    return _normalize_quaternion(q)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.head_pos = nn.Linear(hidden_size, 3)
        self.head_opacity = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor):
        hidden = self.backbone(features)
        return self.head_pos(hidden), self.head_opacity(hidden)


class GaussianBridgeModel(nn.Module):
    def __init__(
        self,
        cfg: BridgeConfig,
        gaussian_xyz0: torch.Tensor,
        gaussian_quat0: torch.Tensor,
        gaussian_scale0: torch.Tensor,
        gaussian_opacity_logits0: torch.Tensor,
        canonical_anchor_pos: torch.Tensor,
        anchor_graph_indices: torch.Tensor,
        gaussian_anchor_indices: torch.Tensor,
        binding_logits_init: torch.Tensor,
        bind_offset_init: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("gaussian_xyz0", gaussian_xyz0)
        self.register_buffer("gaussian_quat0", _normalize_quaternion(gaussian_quat0))
        self.register_buffer("gaussian_scale0", gaussian_scale0)
        self.register_buffer("gaussian_opacity_logits0", gaussian_opacity_logits0)
        self.register_buffer("canonical_anchor_pos", canonical_anchor_pos)
        self.register_buffer("anchor_graph_indices", anchor_graph_indices)
        self.register_buffer("gaussian_anchor_indices", gaussian_anchor_indices)

        self.binding_logits = nn.Parameter(binding_logits_init.clone())
        self.bind_offset = nn.Parameter(bind_offset_init.clone())

        feature_dim = 3 + 3 + 1 + gaussian_xyz0.shape[-1] + gaussian_scale0.shape[-1]
        self.residual_mlp = ResidualMLP(
            input_dim=feature_dim,
            hidden_size=cfg.residual_hidden_size,
            num_layers=cfg.residual_layers,
        )

    def freeze_static_binding(self):
        self.binding_logits.requires_grad_(False)
        self.bind_offset.requires_grad_(False)

    def unfreeze_static_binding(self):
        self.binding_logits.requires_grad_(True)
        self.bind_offset.requires_grad_(True)

    def trainable_stage2_parameters(self):
        return list(self.residual_mlp.parameters())

    def _estimate_anchor_rotations(self, anchors_t: torch.Tensor):
        num_anchors = anchors_t.shape[0]
        canonical = self.canonical_anchor_pos
        neighbors_0 = canonical[self.anchor_graph_indices]
        neighbors_t = anchors_t[self.anchor_graph_indices]
        center_0 = canonical.unsqueeze(1)
        center_t = anchors_t.unsqueeze(1)
        x = neighbors_0 - center_0
        y = neighbors_t - center_t

        cov = torch.einsum("nka,nkb->nab", y, x)
        eye = torch.eye(3, device=anchors_t.device, dtype=anchors_t.dtype).unsqueeze(0)
        rotations = eye.repeat(num_anchors, 1, 1)
        stretch = torch.zeros(num_anchors, device=anchors_t.device, dtype=anchors_t.dtype)
        try:
            u, s, vh = torch.linalg.svd(cov)
            det = torch.det(u @ vh)
            diag = eye.repeat(num_anchors, 1, 1)
            diag[det < 0, -1, -1] = -1.0
            rot = u @ diag @ vh
            valid = torch.isfinite(rot).all(dim=(1, 2))
            rotations[valid] = rot[valid]
            stretch_all = torch.mean(torch.abs(s - 1.0), dim=1)
            stretch[valid] = stretch_all[valid]
        except RuntimeError:
            pass

        quats = _rotation_matrix_to_quaternion(rotations)
        return rotations, quats, stretch

    def forward(
        self,
        anchors_t: torch.Tensor,
        anchors_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = anchors_t.device
        if anchors_prev is None:
            anchors_prev = anchors_t

        rotations, anchor_quats, stretch = self._estimate_anchor_rotations(anchors_t)
        weights = F.softmax(self.binding_logits, dim=-1)
        anchor_idx = self.gaussian_anchor_indices
        selected_anchors = anchors_t[anchor_idx]
        selected_rot = rotations[anchor_idx]
        selected_quat = anchor_quats[anchor_idx]

        local_offsets = self.bind_offset
        rotated_offsets = torch.einsum("gkij,gkj->gki", selected_rot, local_offsets)
        mu_lbs = torch.sum(weights.unsqueeze(-1) * (rotated_offsets + selected_anchors), dim=1)

        quat_blend = torch.sum(weights.unsqueeze(-1) * selected_quat, dim=1)
        quat_blend = _normalize_quaternion(quat_blend)
        quat_out = _quat_mul(quat_blend, self.gaussian_quat0)
        quat_out = _normalize_quaternion(quat_out)

        anchor_vel = anchors_t - anchors_prev
        selected_vel = anchor_vel[anchor_idx]
        weighted_disp = torch.sum(
            weights.unsqueeze(-1) * (selected_anchors - self.canonical_anchor_pos[anchor_idx]), dim=1
        )
        weighted_vel = torch.sum(weights.unsqueeze(-1) * selected_vel, dim=1)
        weighted_stretch = torch.sum(weights * stretch[anchor_idx], dim=1, keepdim=True)

        residual_in = torch.cat(
            [
                weighted_disp,
                weighted_vel,
                weighted_stretch,
                self.gaussian_xyz0,
                self.gaussian_scale0,
            ],
            dim=1,
        )
        delta_x, delta_opacity = self.residual_mlp(residual_in)
        delta_x = torch.tanh(delta_x) * self.cfg.max_pos_residual
        delta_opacity = torch.tanh(delta_opacity) * self.cfg.max_opacity_delta

        xyz = mu_lbs + delta_x
        opacity_logits = self.gaussian_opacity_logits0 + delta_opacity

        return {
            "xyz": xyz,
            "rotation": quat_out,
            "opacity_logits": opacity_logits,
            "weights": weights,
            "delta_x": delta_x,
            "delta_opacity": delta_opacity,
            "stretch": weighted_stretch,
        }

    @staticmethod
    def anchors_from_particles(
        particles_t: torch.Tensor,
        anchor_particle_indices: torch.Tensor,
        anchor_particle_weights: torch.Tensor,
    ) -> torch.Tensor:
        return aggregate_anchor_positions(
            particles_t, anchor_particle_indices, anchor_particle_weights
        )
