from __future__ import annotations

import torch


class DeformedGaussianAdapter:
    """
    Read-only adapter that exposes the runtime-deformed Gaussian state through the same
    property surface the renderer expects from GaussianModel.
    """

    def __init__(
        self,
        base_gaussians,
        xyz: torch.Tensor,
        rotation: torch.Tensor,
        opacity_logits: torch.Tensor,
    ):
        self._base = base_gaussians
        self._xyz_runtime = xyz
        self._rotation_runtime = rotation
        self._opacity_runtime = opacity_logits
        self.active_sh_degree = base_gaussians.active_sh_degree
        self.max_sh_degree = base_gaussians.max_sh_degree
        self.isotropic = base_gaussians.isotropic

    @property
    def get_xyz(self):
        return self._xyz_runtime

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation_runtime, dim=-1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity_runtime)

    @property
    def get_scaling(self):
        scaling = self._base.get_scaling
        if scaling.ndim == 2 and scaling.shape[-1] == 1:
            return scaling.repeat(1, 3)
        return scaling

    @property
    def get_features_dc(self):
        return self._base.get_features_dc

    @property
    def get_features_rest(self):
        return self._base.get_features_rest

    @property
    def get_features(self):
        return self._base.get_features

    def get_normal(self, dir_pp_normalized=None):
        return self._base.get_normal(dir_pp_normalized=dir_pp_normalized)
