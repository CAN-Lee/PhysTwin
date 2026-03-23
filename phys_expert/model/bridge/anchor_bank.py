from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points, sample_farthest_points


@dataclass
class BridgeAssets:
    anchor_particle_indices: torch.Tensor
    anchor_particle_weights: torch.Tensor
    canonical_anchor_pos: torch.Tensor
    anchor_graph_indices: torch.Tensor
    anchor_traj: Optional[torch.Tensor]
    full_particle_count: int
    scene_id: str
    reverse_z: bool
    frame_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "anchor_particle_indices": self.anchor_particle_indices,
            "anchor_particle_weights": self.anchor_particle_weights,
            "canonical_anchor_pos": self.canonical_anchor_pos,
            "anchor_graph_indices": self.anchor_graph_indices,
            "anchor_traj": self.anchor_traj,
            "full_particle_count": self.full_particle_count,
            "scene_id": self.scene_id,
            "reverse_z": self.reverse_z,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "BridgeAssets":
        return cls(
            anchor_particle_indices=data["anchor_particle_indices"],
            anchor_particle_weights=data["anchor_particle_weights"],
            canonical_anchor_pos=data["canonical_anchor_pos"],
            anchor_graph_indices=data["anchor_graph_indices"],
            anchor_traj=data.get("anchor_traj"),
            full_particle_count=int(data["full_particle_count"]),
            scene_id=str(data["scene_id"]),
            reverse_z=bool(data["reverse_z"]),
            frame_count=int(data["frame_count"]),
        )


def _safe_knn(points: torch.Tensor, queries: torch.Tensor, k: int):
    k = max(1, min(k, points.shape[0]))
    dists, indices, _ = knn_points(queries.unsqueeze(0), points.unsqueeze(0), K=k)
    return dists.squeeze(0), indices.squeeze(0)


def _weights_from_sqdist(dist_sq: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    logits = -dist_sq / temperature
    return F.softmax(logits, dim=-1)


def aggregate_anchor_positions(
    particles: torch.Tensor,
    support_indices: torch.Tensor,
    support_weights: torch.Tensor,
) -> torch.Tensor:
    support_pts = particles[support_indices]  # [M, K, 3]
    return torch.sum(support_pts * support_weights.unsqueeze(-1), dim=1)


def build_bridge_assets_from_particles(
    particles_t0: torch.Tensor,
    scene_id: str,
    reverse_z: bool,
    num_render_anchors: int = 4096,
    anchor_particle_k: int = 16,
    anchor_graph_k: int = 8,
    anchor_particle_temp: float = 0.01,
    anchor_traj: Optional[torch.Tensor] = None,
) -> BridgeAssets:
    if particles_t0.ndim != 2 or particles_t0.shape[-1] != 3:
        raise ValueError(f"Expected particles_t0 [N,3], got {tuple(particles_t0.shape)}")

    num_render_anchors = max(1, min(int(num_render_anchors), particles_t0.shape[0]))
    pts_batch = particles_t0.unsqueeze(0)
    try:
        anchor_centers, _ = sample_farthest_points(
            pts_batch,
            K=num_render_anchors,
            random_start_point=False,
        )
    except TypeError:
        anchor_centers, _ = sample_farthest_points(pts_batch, K=num_render_anchors)
    anchor_centers = anchor_centers.squeeze(0)

    dist_sq, anchor_particle_indices = _safe_knn(
        particles_t0, anchor_centers, anchor_particle_k
    )
    anchor_particle_weights = _weights_from_sqdist(dist_sq, anchor_particle_temp)
    canonical_anchor_pos = aggregate_anchor_positions(
        particles_t0, anchor_particle_indices, anchor_particle_weights
    )

    graph_dist_sq, graph_indices = _safe_knn(
        canonical_anchor_pos, canonical_anchor_pos, anchor_graph_k + 1
    )
    # Drop self-neighbor when possible.
    if graph_indices.shape[1] > 1:
        anchor_graph_indices = graph_indices[:, 1:]
    else:
        anchor_graph_indices = graph_indices

    frame_count = int(anchor_traj.shape[0]) if anchor_traj is not None else 0
    return BridgeAssets(
        anchor_particle_indices=anchor_particle_indices.cpu(),
        anchor_particle_weights=anchor_particle_weights.cpu(),
        canonical_anchor_pos=canonical_anchor_pos.cpu(),
        anchor_graph_indices=anchor_graph_indices.cpu(),
        anchor_traj=anchor_traj.cpu() if anchor_traj is not None else None,
        full_particle_count=int(particles_t0.shape[0]),
        scene_id=scene_id,
        reverse_z=reverse_z,
        frame_count=frame_count,
    )


def build_gaussian_anchor_bindings(
    gaussian_xyz: torch.Tensor,
    canonical_anchor_pos: torch.Tensor,
    gaussian_anchor_k: int,
) -> Dict[str, torch.Tensor]:
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[-1] != 3:
        raise ValueError(f"Expected gaussian_xyz [G,3], got {tuple(gaussian_xyz.shape)}")
    if canonical_anchor_pos.ndim != 2 or canonical_anchor_pos.shape[-1] != 3:
        raise ValueError(
            f"Expected canonical_anchor_pos [M,3], got {tuple(canonical_anchor_pos.shape)}"
        )

    dist_sq, anchor_indices = _safe_knn(
        canonical_anchor_pos, gaussian_xyz, gaussian_anchor_k
    )
    init_weights = _weights_from_sqdist(dist_sq, temperature=0.01)
    init_logits = torch.log(init_weights + 1e-8)
    init_bind_offset = gaussian_xyz.unsqueeze(1) - canonical_anchor_pos[anchor_indices]
    return {
        "gaussian_anchor_indices": anchor_indices,
        "binding_logits": init_logits,
        "bind_offset": init_bind_offset,
    }
