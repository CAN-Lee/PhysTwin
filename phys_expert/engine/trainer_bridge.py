from __future__ import annotations

import copy
import json
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.loss_utils import l1_loss, ssim

from ..data.dataset_bridge import BridgeSequenceDataset
from ..model.bridge import (
    BridgeAssets,
    BridgeConfig,
    DeformedGaussianAdapter,
    GaussianBridgeModel,
    build_gaussian_anchor_bindings,
)
from .trainer_mpm import PhysExpertMPMTrainer


def resolve_gaussian_ply_path(cfg, scene_id: str) -> str:
    root = getattr(cfg.bridge, "gaussian_root", "./gaussian_output")
    exp_name = getattr(cfg.bridge, "gaussian_exp_name", "auto_latest")
    scene_root = os.path.join(root, scene_id)
    if not os.path.isdir(scene_root):
        raise FileNotFoundError(f"Gaussian root not found for scene {scene_id}: {scene_root}")

    if exp_name and exp_name != "auto_latest":
        exp_dirs = [os.path.join(scene_root, exp_name)]
    else:
        exp_dirs = [
            os.path.join(scene_root, d)
            for d in os.listdir(scene_root)
            if os.path.isdir(os.path.join(scene_root, d))
        ]
        exp_dirs.sort(key=os.path.getmtime, reverse=True)

    for exp_dir in exp_dirs:
        point_cloud_dir = os.path.join(exp_dir, "point_cloud")
        if not os.path.isdir(point_cloud_dir):
            continue
        iters = [
            os.path.join(point_cloud_dir, d, "point_cloud.ply")
            for d in os.listdir(point_cloud_dir)
            if d.startswith("iteration_")
        ]
        iters = [p for p in iters if os.path.exists(p)]
        if not iters:
            continue
        iters.sort(key=os.path.getmtime, reverse=True)
        return iters[0]
    raise FileNotFoundError(f"No Gaussian point_cloud.ply found under {scene_root}")


def load_bridge_assets(path: str) -> BridgeAssets:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bridge assets not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return BridgeAssets.from_dict(data)


def build_bridge_config(cfg) -> BridgeConfig:
    bridge_cfg = getattr(cfg, "bridge", {})
    return BridgeConfig(
        gaussian_anchor_k=int(getattr(bridge_cfg, "gaussian_anchor_k", 8)),
        anchor_graph_k=int(getattr(bridge_cfg, "anchor_graph_k", 8)),
        max_pos_residual=float(getattr(bridge_cfg, "max_pos_residual", 0.005)),
        max_opacity_delta=float(getattr(bridge_cfg, "max_opacity_delta", 0.5)),
        residual_hidden_size=int(getattr(bridge_cfg, "residual_hidden_size", 128)),
        residual_layers=int(getattr(bridge_cfg, "residual_layers", 3)),
    )


@dataclass
class RenderLossOutput:
    total: torch.Tensor
    rgb: torch.Tensor
    mask: torch.Tensor
    bind: torch.Tensor
    sparse: torch.Tensor
    res_pos: torch.Tensor
    res_opacity: torch.Tensor
    temporal: torch.Tensor
    canonical_xyz: torch.Tensor
    canonical_opacity: torch.Tensor


class SceneBridgeBundle:
    def __init__(
        self,
        cfg,
        case_name: str,
        device: torch.device,
        bridge_assets_path: Optional[str] = None,
        bridge_checkpoint: Optional[str] = None,
    ):
        self.cfg = cfg
        self.case_name = case_name
        self.device = device
        self.bridge_cfg = cfg.bridge

        checkpoint_payload = None
        if bridge_checkpoint and os.path.exists(bridge_checkpoint):
            checkpoint_payload = torch.load(
                bridge_checkpoint,
                map_location="cpu",
                weights_only=False,
            )

        if bridge_assets_path is None:
            if checkpoint_payload is not None:
                checkpoint_assets_path = checkpoint_payload.get("bridge_assets_path")
                if checkpoint_assets_path and os.path.exists(checkpoint_assets_path):
                    bridge_assets_path = checkpoint_assets_path
            if bridge_assets_path is None:
                inference_dir = getattr(self.bridge_cfg, "inference_dir", "./output_3/mpm_inference")
                bridge_assets_path = os.path.join(inference_dir, case_name, "bridge_assets.pt")
        self.bridge_assets_path = bridge_assets_path
        self.assets = load_bridge_assets(bridge_assets_path)

        self.train_dataset = BridgeSequenceDataset(cfg, case_name=case_name, split="train")
        self.test_dataset = BridgeSequenceDataset(cfg, case_name=case_name, split="test")
        self.num_cams = self.train_dataset.num_cams
        self.train_frame_set = set(self.train_dataset.frame_indices)
        self.test_frame_set = set(self.test_dataset.frame_indices)

        self.gaussian_ply_path = resolve_gaussian_ply_path(cfg, case_name)
        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.load_ply(self.gaussian_ply_path)
        loaded_isotropic = (
            hasattr(self.gaussians, "_scaling")
            and self.gaussians._scaling is not None
            and self.gaussians._scaling.ndim == 2
            and self.gaussians._scaling.shape[1] == 1
        )
        self.gaussians.isotropic = getattr(self.bridge_cfg, "isotropic", loaded_isotropic)
        for attr in [
            self.gaussians._xyz,
            self.gaussians._features_dc,
            self.gaussians._features_rest,
            self.gaussians._scaling,
            self.gaussians._rotation,
            self.gaussians._opacity,
        ]:
            attr.requires_grad_(False)

        bindings = build_gaussian_anchor_bindings(
            gaussian_xyz=self.gaussians.get_xyz.detach().cpu(),
            canonical_anchor_pos=self.assets.canonical_anchor_pos,
            gaussian_anchor_k=int(getattr(self.bridge_cfg, "gaussian_anchor_k", 8)),
        )
        self.bridge_model = GaussianBridgeModel(
            cfg=build_bridge_config(cfg),
            gaussian_xyz0=self.gaussians.get_xyz.detach(),
            gaussian_quat0=self.gaussians.get_rotation.detach(),
            gaussian_scale0=self.gaussians.get_scaling.detach(),
            gaussian_opacity_logits0=self.gaussians._opacity.detach(),
            canonical_anchor_pos=self.assets.canonical_anchor_pos.to(device),
            anchor_graph_indices=self.assets.anchor_graph_indices.to(device),
            gaussian_anchor_indices=bindings["gaussian_anchor_indices"].to(device),
            binding_logits_init=bindings["binding_logits"].to(device),
            bind_offset_init=bindings["bind_offset"].to(device),
        ).to(device)
        self.bind_offset_init = bindings["bind_offset"].to(device)

        if checkpoint_payload is not None:
            state = checkpoint_payload.get("bridge_model_state_dict", checkpoint_payload)
            self.bridge_model.load_state_dict(state, strict=False)

        self.anchor_particle_indices = self.assets.anchor_particle_indices.to(device)
        self.anchor_particle_weights = self.assets.anchor_particle_weights.to(device)
        self.canonical_anchor_pos = self.assets.canonical_anchor_pos.to(device)
        self.anchor_traj = (
            self.assets.anchor_traj.to(device) if self.assets.anchor_traj is not None else None
        )

        bg = getattr(self.bridge_cfg, "background", [0.0, 0.0, 0.0])
        self.background = torch.tensor(bg, dtype=torch.float32, device=device)

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, object]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bridge_assets_snapshot = os.path.join(os.path.dirname(path), "bridge_assets.pt")
        source_assets_path = self.bridge_assets_path
        if source_assets_path and os.path.exists(source_assets_path):
            if os.path.abspath(source_assets_path) != os.path.abspath(bridge_assets_snapshot):
                shutil.copy2(source_assets_path, bridge_assets_snapshot)
            else:
                bridge_assets_snapshot = source_assets_path
        payload = {
            "bridge_model_state_dict": self.bridge_model.state_dict(),
            "gaussian_ply_path": self.gaussian_ply_path,
            "bridge_assets_path": bridge_assets_snapshot,
            "scene_id": self.case_name,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def sample_dataset(self, split: str = "train"):
        dataset = self.train_dataset if split == "train" else self.test_dataset
        return dataset[random.randrange(len(dataset))]

    def get_dataset_sample(self, frame_idx: int, cam_idx: int, split: str = "train"):
        dataset = self.train_dataset if split == "train" else self.test_dataset
        return dataset.get_sample(frame_idx, cam_idx)

    def get_frame_sample(self, frame_idx: int, cam_idx: int):
        if frame_idx in self.train_frame_set:
            return self.train_dataset.get_sample(frame_idx, cam_idx)
        if frame_idx in self.test_frame_set:
            return self.test_dataset.get_sample(frame_idx, cam_idx)
        raise KeyError(f"Frame {frame_idx} not found in train/test split for scene {self.case_name}")

    def anchors_from_particles(self, particles_world: torch.Tensor) -> torch.Tensor:
        return self.bridge_model.anchors_from_particles(
            particles_world, self.anchor_particle_indices, self.anchor_particle_weights
        )

    def deform_from_anchors(self, anchors_t: torch.Tensor, anchors_prev: Optional[torch.Tensor] = None):
        return self.bridge_model(anchors_t=anchors_t, anchors_prev=anchors_prev)

    def deform_from_particles(
        self,
        particles_world: torch.Tensor,
        prev_particles_world: Optional[torch.Tensor] = None,
    ):
        anchors_t = self.anchors_from_particles(particles_world)
        anchors_prev = (
            self.anchors_from_particles(prev_particles_world)
            if prev_particles_world is not None
            else anchors_t
        )
        outputs = self.deform_from_anchors(anchors_t, anchors_prev=anchors_prev)
        outputs["anchors_t"] = anchors_t
        outputs["anchors_prev"] = anchors_prev
        return outputs

    def build_adapter(self, outputs: Dict[str, torch.Tensor]) -> DeformedGaussianAdapter:
        return DeformedGaussianAdapter(
            base_gaussians=self.gaussians,
            xyz=outputs["xyz"],
            rotation=outputs["rotation"],
            opacity_logits=outputs["opacity_logits"],
        )

    def _mask_valid(self, sample) -> torch.Tensor:
        valid = sample.camera.alpha_mask
        if sample.camera.occ_mask is not None:
            valid = valid * (1.0 - sample.camera.occ_mask.unsqueeze(0))
        return valid.clamp(0.0, 1.0)

    def compute_render_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        sample,
        prev_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> RenderLossOutput:
        adapter = self.build_adapter(outputs)
        results = render(sample.camera, adapter, None, self.background)
        pred_rgba = results["render"]
        pred_rgb = pred_rgba[:3]
        pred_alpha = pred_rgba[3:4]

        gt_rgb = sample.camera.original_image
        gt_alpha = sample.camera.alpha_mask
        valid_mask = self._mask_valid(sample)

        pred_rgb_valid = pred_rgb * valid_mask
        gt_rgb_valid = gt_rgb * valid_mask
        rgb_loss = (1.0 - float(getattr(self.bridge_cfg, "lambda_dssim", 0.2))) * l1_loss(
            pred_rgb_valid, gt_rgb_valid
        )
        rgb_loss = rgb_loss + float(getattr(self.bridge_cfg, "lambda_dssim", 0.2)) * (
            1.0 - ssim(pred_rgb_valid, gt_rgb_valid)
        )
        mask_loss = l1_loss(pred_alpha * valid_mask, gt_alpha * valid_mask)

        weights = outputs["weights"]
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        bind_loss = F.mse_loss(self.bridge_model.bind_offset, self.bind_offset_init)
        res_pos = torch.mean(outputs["delta_x"] ** 2)
        res_opacity = torch.mean(outputs["delta_opacity"] ** 2)
        temporal = torch.tensor(0.0, device=self.device)
        canonical_xyz = torch.tensor(0.0, device=self.device)
        canonical_opacity = torch.tensor(0.0, device=self.device)
        if prev_outputs is not None:
            temporal = F.mse_loss(outputs["delta_x"], prev_outputs["delta_x"].detach())
        if sample.frame_idx == 0:
            canonical_xyz = F.mse_loss(outputs["xyz"], self.bridge_model.gaussian_xyz0)
            canonical_opacity = F.mse_loss(
                outputs["opacity_logits"], self.bridge_model.gaussian_opacity_logits0
            )

        total = (
            float(getattr(self.bridge_cfg, "lambda_rgb", 1.0)) * rgb_loss
            + float(getattr(self.bridge_cfg, "lambda_mask", 0.5)) * mask_loss
            + float(getattr(self.bridge_cfg, "lambda_bind", 1e-4)) * bind_loss
            + float(getattr(self.bridge_cfg, "lambda_weight_sparse", 1e-4)) * entropy
            + float(getattr(self.bridge_cfg, "lambda_res_pos", 1e-4)) * res_pos
            + float(getattr(self.bridge_cfg, "lambda_res_opacity", 1e-4)) * res_opacity
            + float(getattr(self.bridge_cfg, "lambda_res_temporal", 1e-4)) * temporal
            + float(getattr(self.bridge_cfg, "lambda_canonical_xyz", 0.0)) * canonical_xyz
            + float(getattr(self.bridge_cfg, "lambda_canonical_opacity", 0.0)) * canonical_opacity
        )
        return RenderLossOutput(
            total=total,
            rgb=rgb_loss,
            mask=mask_loss,
            bind=bind_loss,
            sparse=entropy,
            res_pos=res_pos,
            res_opacity=res_opacity,
            temporal=temporal,
            canonical_xyz=canonical_xyz,
            canonical_opacity=canonical_opacity,
        )


class BridgeTrainer:
    def __init__(self, cfg, case_name: str, inference_dir: str = "./output_3/mpm_inference"):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.bridge.inference_dir = inference_dir
        self.case_name = case_name
        self.device = torch.device(getattr(cfg.mpm, "device", "cuda"))
        self.bundle = SceneBridgeBundle(self.cfg, case_name, self.device)
        self.log_dir = os.path.join(cfg.output_dir, case_name, "bridge_stage1")
        os.makedirs(self.log_dir, exist_ok=True)
        if self.bundle.anchor_traj is None:
            raise ValueError("Stage-1 bridge training requires bridge_assets.pt with anchor_traj")
        train_static_binding = bool(getattr(self.cfg.bridge, "stage1_train_static_binding", False))
        train_residual = bool(getattr(self.cfg.bridge, "stage1_train_residual", True))

        if not train_static_binding:
            self.bundle.bridge_model.freeze_static_binding()

        param_groups = []
        if train_static_binding:
            param_groups.append(
                {
                    "params": [self.bundle.bridge_model.binding_logits, self.bundle.bridge_model.bind_offset],
                    "lr": float(getattr(self.cfg.bridge, "stage1_lr_static", 1e-3)),
                }
            )
        if train_residual:
            param_groups.append(
                {
                    "params": self.bundle.bridge_model.residual_mlp.parameters(),
                    "lr": float(getattr(self.cfg.bridge, "stage1_lr_residual", 1e-3)),
                }
            )
        if not param_groups:
            raise ValueError("Stage-1 bridge training has no enabled parameter groups")

        self.optimizer = torch.optim.Adam(param_groups)
        self.best_loss = float("inf")
        self.best_val_loss = float("inf")

    def _anchors_for_frame(self, frame_idx: int):
        anchors_t = self.bundle.anchor_traj[frame_idx]
        prev_idx = max(frame_idx - 1, 0)
        anchors_prev = self.bundle.anchor_traj[prev_idx]
        return anchors_t, anchors_prev

    def _forward_frame(self, frame_idx: int):
        anchors_t, anchors_prev = self._anchors_for_frame(frame_idx)
        outputs = self.bundle.deform_from_anchors(anchors_t, anchors_prev)
        prev_outputs = None
        if frame_idx > 0:
            prev_anchors_t, prev_anchors_prev = self._anchors_for_frame(frame_idx - 1)
            prev_outputs = self.bundle.deform_from_anchors(prev_anchors_t, prev_anchors_prev)
        return outputs, prev_outputs

    def validate(self, max_samples: Optional[int] = None) -> float:
        self.bundle.bridge_model.eval()
        losses = []
        with torch.no_grad():
            dataset = self.bundle.test_dataset if len(self.bundle.test_dataset) > 0 else self.bundle.train_dataset
            if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
                indices = list(range(len(dataset)))
            else:
                step = max(1, len(dataset) // max_samples)
                indices = list(range(0, len(dataset), step))[:max_samples]
            for idx in indices:
                sample = dataset[idx]
                outputs, prev_outputs = self._forward_frame(sample.frame_idx)
                loss = self.bundle.compute_render_loss(outputs, sample, prev_outputs)
                losses.append(loss.total.item())
        self.bundle.bridge_model.train()
        return float(sum(losses) / max(1, len(losses)))

    def _train_epoch_global(self) -> float:
        dataset = self.bundle.train_dataset
        sample_indices = list(range(len(dataset)))
        if bool(getattr(self.cfg.bridge, "stage1_shuffle_train_samples", True)):
            random.shuffle(sample_indices)

        num_samples = max(1, len(sample_indices))
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for sample_idx in sample_indices:
            sample = dataset[sample_idx]
            outputs, prev_outputs = self._forward_frame(sample.frame_idx)
            losses = self.bundle.compute_render_loss(outputs, sample, prev_outputs)
            (losses.total / num_samples).backward()
            total_loss += losses.total.item()

        torch.nn.utils.clip_grad_norm_(
            self.bundle.bridge_model.parameters(),
            float(getattr(self.cfg.bridge, "stage1_grad_clip_norm", 1.0)),
        )
        self.optimizer.step()
        return total_loss / num_samples

    def train(self, num_iters: Optional[int] = None):
        num_iters = int(num_iters or getattr(self.cfg.bridge, "stage1_iters", 20000))
        eval_every = int(getattr(self.cfg.bridge, "eval_every", 500))
        val_max_samples = int(getattr(self.cfg.bridge, "stage1_val_max_samples", 0))
        global_opt = bool(getattr(self.cfg.bridge, "stage1_global_optimization", True))

        for step in range(num_iters):
            if global_opt:
                train_loss = self._train_epoch_global()
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.bundle.save_checkpoint(
                        os.path.join(self.log_dir, "best_bridge_train.pt"),
                        extra={"iter": step + 1, "best_train_loss": self.best_loss},
                    )
            else:
                self.optimizer.zero_grad(set_to_none=True)
                sample = self.bundle.sample_dataset(split="train")
                outputs, prev_outputs = self._forward_frame(sample.frame_idx)
                losses = self.bundle.compute_render_loss(outputs, sample, prev_outputs)
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.bundle.bridge_model.parameters(),
                    float(getattr(self.cfg.bridge, "stage1_grad_clip_norm", 1.0)),
                )
                self.optimizer.step()
                train_loss = losses.total.item()
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.bundle.save_checkpoint(
                        os.path.join(self.log_dir, "best_bridge_train.pt"),
                        extra={"iter": step + 1, "best_train_loss": self.best_loss},
                    )

            if (step + 1) % eval_every == 0 or (step + 1) == num_iters:
                val_loss = self.validate(max_samples=val_max_samples)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.bundle.save_checkpoint(
                        os.path.join(self.log_dir, "best_bridge.pt"),
                        extra={"iter": step + 1, "best_val_loss": self.best_val_loss},
                    )
                print(
                    f"[Bridge][{self.case_name}] iter={step+1}/{num_iters} "
                    f"train={train_loss:.6f} val={val_loss:.6f}"
                )

        final_path = os.path.join(self.log_dir, "final_bridge.pt")
        self.bundle.save_checkpoint(
            final_path,
            extra={
                "iter": num_iters,
                "best_train_loss": self.best_loss,
                "best_val_loss": self.best_val_loss,
            },
        )
        return final_path


class JointBridgeMPMTrainer(PhysExpertMPMTrainer):
    def __init__(
        self,
        cfg,
        scene_id: str,
        bridge_checkpoint: str,
        inference_dir: str = "./output_3/mpm_inference",
        resume_path: Optional[str] = None,
    ):
        self._joint_bridge_checkpoint = bridge_checkpoint
        self._joint_inference_dir = inference_dir
        self._joint_prev_particles_world = None
        super().__init__(cfg, scene_id, resume_path=resume_path)

        self.cfg.bridge.inference_dir = inference_dir
        self.bridge_bundle = SceneBridgeBundle(
            self.cfg,
            scene_id,
            self.device,
            bridge_checkpoint=bridge_checkpoint,
        )
        self.bridge_bundle.bridge_model.freeze_static_binding()
        self.optimizer.add_param_group(
            {
                "params": self.bridge_bundle.bridge_model.trainable_stage2_parameters(),
                "lr": float(getattr(self.cfg.bridge, "stage2_lr", 1e-4)),
            }
        )
        # Joint finetune warm-starts from the physics weights but should run its own
        # short schedule from iter 0 instead of resuming the original training loop.
        self.resume_checkpoint = None

        split_path = os.path.join(self.cfg.data.root, scene_id, "split.json")
        with open(split_path, "r") as f:
            split = json.load(f)
        start, end = split["train"]
        stride = int(getattr(self.cfg.bridge, "render_frame_stride", 10))
        self.render_supervised_frames = {idx for idx in range(start, end) if idx % stride == 0}

    def _particles_to_world(self, particles_shifted: torch.Tensor) -> torch.Tensor:
        world = particles_shifted - self.auto_offset
        if getattr(self.cfg.data, "reverse_z", False):
            world = world.clone()
            world[..., 2] *= -1.0
        return world

    def on_train_iteration_start(self, iter_idx: int, total_frames: int):
        world_init = self._particles_to_world((self.simulator.x - self.simulator.shift).detach())
        self._joint_prev_particles_world = world_init

    def compute_optional_render_loss(self, iter_idx: int, frame_idx: int, x_curr: torch.Tensor):
        if frame_idx not in self.render_supervised_frames:
            self._joint_prev_particles_world = self._particles_to_world(x_curr.detach())
            return torch.tensor(0.0, device=self.device)

        cam_idx = (iter_idx + frame_idx) % self.bridge_bundle.num_cams
        sample = self.bridge_bundle.get_dataset_sample(frame_idx, cam_idx, split="train")
        curr_world = self._particles_to_world(x_curr)
        prev_world = self._joint_prev_particles_world
        outputs = self.bridge_bundle.deform_from_particles(curr_world, prev_particles_world=prev_world)
        losses = self.bridge_bundle.compute_render_loss(outputs, sample)

        warmup_iters = int(getattr(self.cfg.bridge, "joint_warmup_iters", 10))
        target = float(getattr(self.cfg.bridge, "lambda_render_joint", 0.05))
        if warmup_iters > 0:
            warmup_scale = min(1.0, float(iter_idx + 1) / float(warmup_iters))
        else:
            warmup_scale = 1.0
        self._joint_prev_particles_world = curr_world.detach()
        return losses.total * target * warmup_scale
