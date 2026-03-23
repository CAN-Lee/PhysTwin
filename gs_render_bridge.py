import argparse
import os

import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from phys_expert.engine.trainer_bridge import SceneBridgeBundle


def main():
    parser = argparse.ArgumentParser(description="Render dynamic Gaussians with learned bridge")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bridge_checkpoint", type=str, required=True)
    parser.add_argument("--inference_dir", type=str, default="./output_3/mpm_inference")
    parser.add_argument("--output_dir", type=str, default="./gaussian_output_bridge")
    parser.add_argument("--views", type=str, default="0,1,2")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.mpm.device = "cuda"
    cfg.bridge.inference_dir = args.inference_dir

    bundle = SceneBridgeBundle(
        cfg,
        args.case_name,
        device=torch.device("cuda"),
        bridge_checkpoint=args.bridge_checkpoint,
    )
    if bundle.anchor_traj is None:
        raise ValueError("bridge_assets.pt must include anchor_traj for rendering")

    view_indices = [int(v) for v in args.views.split(",") if v.strip()]
    base_out = os.path.join(args.output_dir, args.case_name)
    os.makedirs(base_out, exist_ok=True)
    for view_idx in view_indices:
        os.makedirs(os.path.join(base_out, str(view_idx)), exist_ok=True)

    with torch.no_grad():
        for frame_idx in tqdm(range(bundle.anchor_traj.shape[0]), desc=f"Render {args.case_name}"):
            anchors_t = bundle.anchor_traj[frame_idx]
            anchors_prev = bundle.anchor_traj[max(frame_idx - 1, 0)]
            outputs = bundle.deform_from_anchors(anchors_t, anchors_prev)
            adapter = bundle.build_adapter(outputs)
            for view_idx in view_indices:
                sample = bundle.get_frame_sample(frame_idx, view_idx)
                results = render(sample.camera, adapter, None, bundle.background)
                out_path = os.path.join(base_out, str(view_idx), f"{frame_idx:05d}.png")
                torchvision.utils.save_image(results["render"], out_path)


if __name__ == "__main__":
    main()
