"""
Batch inference script for the Hybrid Physics-Neural MPM Simulator.

Iterates over all scenes in output_3/*_warp directories, runs inference
using best_checkpoint.pt, and exports inference.pkl to a flat directory
structure: output_3/mpm_inference/{case_name}/inference.pkl

This flat layout is compatible with gs_render_dynamics.py's --prediction_dir.

Usage:
    python script_inference_mpm.py [--gpu 0]
    python script_inference_mpm.py --gpu 0 --scenes double_lift_sloth,double_lift_zebra
"""

import os
import sys
import argparse
import subprocess


# Scene category -> config mapping
CATEGORY_CONFIGS = {
    'softbody_warp': 'configs/softbody.yaml',
    'cloth_warp': 'configs/mpm_cloth.yaml',
    'rope_warp': 'configs/rope.yaml',
}

OUTPUT_DIR = './output_3/mpm_inference'


def discover_scenes():
    """Auto-discover all scenes from output_3/*_warp directories."""
    scenes = []
    for category, config in CATEGORY_CONFIGS.items():
        cat_dir = os.path.join('./output_3', category)
        if not os.path.isdir(cat_dir):
            continue
        for scene_name in sorted(os.listdir(cat_dir)):
            scene_dir = os.path.join(cat_dir, scene_name)
            ckpt_path = os.path.join(scene_dir, 'best_checkpoint.pt')
            if os.path.isdir(scene_dir) and os.path.exists(ckpt_path):
                scenes.append((scene_name, config, ckpt_path))
            else:
                print(f"  Skipping {scene_name} (no best_checkpoint.pt)")
    return scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene names to process (default: all)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    all_scenes = discover_scenes()
    print(f"Discovered {len(all_scenes)} scenes with best_checkpoint.pt")

    if args.scenes:
        filter_set = set(args.scenes.split(','))
        all_scenes = [(n, c, p) for n, c, p in all_scenes if n in filter_set]
        print(f"Filtered to {len(all_scenes)} scenes: {[s[0] for s in all_scenes]}")

    for i, (scene_name, config, ckpt_path) in enumerate(all_scenes):
        save_path = os.path.join(args.output_dir, scene_name, 'inference.pkl')
        bridge_path = os.path.join(args.output_dir, scene_name, 'bridge_assets.pt')
        if os.path.exists(save_path) and os.path.exists(bridge_path):
            print(f"[{i+1}/{len(all_scenes)}] {scene_name}: inference.pkl and bridge_assets.pt exist, skipping.")
            continue

        print(f"\n[{i+1}/{len(all_scenes)}] Processing {scene_name} ...")
        cmd = [
            sys.executable, "inference_mpm.py",
            "--case_name", scene_name,
            "--config", config,
            "--checkpoint", ckpt_path,
            "--output_dir", args.output_dir,
            "--gpu", args.gpu,
            "--save_bridge_assets",
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"  -> Done: {scene_name}")
        except subprocess.CalledProcessError:
            print(f"  -> FAILED: {scene_name}")

    print(f"\nAll done. Inference results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
