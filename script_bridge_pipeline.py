"""
Run the full bridge pipeline for a single scene.

Stages:
1. Export `inference.pkl` and `bridge_assets.pt`
2. Train the stage-1 bridge
3. Jointly finetune physics + bridge residual head
4. Render the final dynamic Gaussian sequence

By default, the render stage refreshes inference with the final joint physics
checkpoint so the rendered sequence matches the latest physics rollout.
"""

import argparse
import os
import subprocess
import sys


def _run_step(step_name, cmd, env):
    print(f"\n=== {step_name} ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def _default_bridge_output_root(base_output_root):
    return f"{base_output_root}_bridge"


def _default_physics_checkpoint(base_output_root, case_name):
    return os.path.join(base_output_root, case_name, "best_checkpoint.pt")


def _default_stage1_bridge_checkpoint(bridge_output_root, case_name):
    return os.path.join(bridge_output_root, case_name, "bridge_stage1", "best_bridge.pt")


def _default_joint_physics_checkpoint(bridge_output_root, case_name):
    return os.path.join(bridge_output_root, case_name, "final_checkpoint.pt")


def _default_joint_bridge_checkpoint(bridge_output_root, case_name):
    return os.path.join(bridge_output_root, case_name, "bridge_stage2", "final_bridge_joint.pt")


def main():
    parser = argparse.ArgumentParser(description="Run the 4-step bridge training/render pipeline")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--bridge_output_root", type=str, default=None)
    parser.add_argument("--physics_checkpoint", type=str, default=None)
    parser.add_argument("--bridge_checkpoint", type=str, default=None)
    parser.add_argument("--joint_physics_checkpoint", type=str, default=None)
    parser.add_argument("--joint_bridge_checkpoint", type=str, default=None)
    parser.add_argument("--inference_dir", type=str, default=None)
    parser.add_argument("--render_output_dir", type=str, default=None)
    parser.add_argument("--views", type=str, default="0,1,2")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--no_refresh_inference_after_joint", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    base_output_root = cfg.output_dir
    bridge_output_root = args.bridge_output_root or _default_bridge_output_root(base_output_root)
    inference_dir = args.inference_dir or os.path.join(bridge_output_root, "inference")
    render_output_dir = args.render_output_dir or os.path.join(bridge_output_root, "renders")

    physics_checkpoint = args.physics_checkpoint or _default_physics_checkpoint(
        base_output_root, args.case_name
    )
    stage1_bridge_checkpoint = args.bridge_checkpoint or _default_stage1_bridge_checkpoint(
        bridge_output_root, args.case_name
    )
    joint_physics_checkpoint = args.joint_physics_checkpoint or _default_joint_physics_checkpoint(
        bridge_output_root, args.case_name
    )
    joint_bridge_checkpoint = args.joint_bridge_checkpoint or _default_joint_bridge_checkpoint(
        bridge_output_root, args.case_name
    )

    inference_pkl = os.path.join(inference_dir, args.case_name, "inference.pkl")
    bridge_assets = os.path.join(inference_dir, args.case_name, "bridge_assets.pt")
    refresh_after_joint = not args.no_refresh_inference_after_joint

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env_python_bin = os.path.dirname(sys.executable)
    env["PATH"] = env_python_bin + os.pathsep + env.get("PATH", "")

    print("Case:", args.case_name)
    print("Config:", args.config)
    print("GPU:", args.gpu)
    print("Bridge output root:", bridge_output_root)
    print("Physics checkpoint:", physics_checkpoint)
    print("Stage-1 bridge checkpoint:", stage1_bridge_checkpoint)
    print("Joint physics checkpoint:", joint_physics_checkpoint)
    print("Joint bridge checkpoint:", joint_bridge_checkpoint)

    if not os.path.exists(physics_checkpoint):
        raise FileNotFoundError(f"Physics checkpoint not found: {physics_checkpoint}")

    if not args.skip_inference:
        if args.force or not (os.path.exists(inference_pkl) and os.path.exists(bridge_assets)):
            cmd = [
                sys.executable,
                "inference_mpm.py",
                "--case_name",
                args.case_name,
                "--config",
                args.config,
                "--checkpoint",
                physics_checkpoint,
                "--output_dir",
                inference_dir,
                "--gpu",
                args.gpu,
                "--save_bridge_assets",
            ]
            _run_step("Step 1/4: Physics Inference + Bridge Assets", cmd, env)
        else:
            print("\n=== Step 1/4: Physics Inference + Bridge Assets ===")
            print("Skip: existing inference.pkl and bridge_assets.pt found.")

    if not args.skip_stage1:
        if args.force or not os.path.exists(stage1_bridge_checkpoint):
            cmd = [
                sys.executable,
                "train_bridge.py",
                "--case_name",
                args.case_name,
                "--config",
                args.config,
                "--stage",
                "bridge",
                "--output_root",
                bridge_output_root,
                "--inference_dir",
                inference_dir,
            ]
            _run_step("Step 2/4: Train Bridge", cmd, env)
        else:
            print("\n=== Step 2/4: Train Bridge ===")
            print(f"Skip: existing checkpoint found at {stage1_bridge_checkpoint}")

    if not args.skip_stage2:
        if not os.path.exists(stage1_bridge_checkpoint):
            raise FileNotFoundError(
                f"Stage-1 bridge checkpoint not found: {stage1_bridge_checkpoint}"
            )
        if args.force or not (
            os.path.exists(joint_physics_checkpoint) and os.path.exists(joint_bridge_checkpoint)
        ):
            cmd = [
                sys.executable,
                "train_bridge.py",
                "--case_name",
                args.case_name,
                "--config",
                args.config,
                "--stage",
                "joint",
                "--output_root",
                bridge_output_root,
                "--physics_checkpoint",
                physics_checkpoint,
                "--bridge_checkpoint",
                stage1_bridge_checkpoint,
                "--joint_bridge_output",
                joint_bridge_checkpoint,
                "--inference_dir",
                inference_dir,
            ]
            _run_step("Step 3/4: Joint Finetune Physics + Bridge", cmd, env)
        else:
            print("\n=== Step 3/4: Joint Finetune Physics + Bridge ===")
            print(
                "Skip: existing joint physics and joint bridge checkpoints found."
            )

    render_bridge_checkpoint = (
        joint_bridge_checkpoint
        if os.path.exists(joint_bridge_checkpoint)
        else stage1_bridge_checkpoint
    )
    render_physics_checkpoint = (
        joint_physics_checkpoint
        if os.path.exists(joint_physics_checkpoint)
        else physics_checkpoint
    )

    if not args.skip_render:
        if not os.path.exists(render_bridge_checkpoint):
            raise FileNotFoundError(
                f"Render bridge checkpoint not found: {render_bridge_checkpoint}"
            )

        should_refresh_inference = (
            refresh_after_joint
            and os.path.exists(render_physics_checkpoint)
            and (
                not (os.path.exists(inference_pkl) and os.path.exists(bridge_assets))
                or os.path.abspath(render_physics_checkpoint) != os.path.abspath(physics_checkpoint)
            )
        )
        if should_refresh_inference:
            cmd = [
                sys.executable,
                "inference_mpm.py",
                "--case_name",
                args.case_name,
                "--config",
                args.config,
                "--checkpoint",
                render_physics_checkpoint,
                "--output_dir",
                inference_dir,
                "--gpu",
                args.gpu,
                "--save_bridge_assets",
            ]
            _run_step("Step 4/4a: Refresh Inference For Final Render", cmd, env)
        elif refresh_after_joint:
            print("\n=== Step 4/4a: Refresh Inference For Final Render ===")
            print("Skip: existing inference already matches the render physics checkpoint.")

        cmd = [
            sys.executable,
            "gs_render_bridge.py",
            "--case_name",
            args.case_name,
            "--config",
            args.config,
            "--bridge_checkpoint",
            render_bridge_checkpoint,
            "--inference_dir",
            inference_dir,
            "--output_dir",
            render_output_dir,
            "--views",
            args.views,
        ]
        _run_step("Step 4/4b: Render Final Dynamic GS", cmd, env)

    print("\nPipeline finished.")
    print("Final render bridge checkpoint:", render_bridge_checkpoint)
    print("Final render physics checkpoint:", render_physics_checkpoint)
    print("Inference dir:", os.path.join(inference_dir, args.case_name))
    print("Render dir:", os.path.join(render_output_dir, args.case_name))


if __name__ == "__main__":
    main()
