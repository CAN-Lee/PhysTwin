import argparse
import os

from omegaconf import OmegaConf

from phys_expert.engine.trainer_bridge import BridgeTrainer, JointBridgeMPMTrainer


def default_physics_checkpoint(cfg, case_name: str) -> str:
    return os.path.join(cfg.output_dir, case_name, "best_checkpoint.pt")


def default_bridge_checkpoint(cfg, case_name: str) -> str:
    return os.path.join(cfg.output_dir, case_name, "bridge_stage1", "best_bridge.pt")


def main():
    parser = argparse.ArgumentParser(description="Train differentiable particle-anchor-GS bridge")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default="bridge", choices=["bridge", "joint"])
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--physics_checkpoint", type=str, default=None)
    parser.add_argument("--bridge_checkpoint", type=str, default=None)
    parser.add_argument("--joint_bridge_output", type=str, default=None)
    parser.add_argument("--inference_dir", type=str, default="./output_3/mpm_inference")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.mpm.device = "cuda"
    base_output_root = cfg.output_dir
    run_output_root = args.output_root or base_output_root
    cfg.output_dir = run_output_root

    if args.stage == "bridge":
        trainer = BridgeTrainer(cfg, args.case_name, inference_dir=args.inference_dir)
        trainer.train(num_iters=getattr(cfg.bridge, "stage1_iters", 20000))
        return

    physics_checkpoint = args.physics_checkpoint or os.path.join(
        base_output_root, args.case_name, "best_checkpoint.pt"
    )
    bridge_checkpoint = args.bridge_checkpoint or default_bridge_checkpoint(cfg, args.case_name)
    if not os.path.exists(physics_checkpoint):
        raise FileNotFoundError(f"Physics checkpoint not found: {physics_checkpoint}")
    if not os.path.exists(bridge_checkpoint):
        raise FileNotFoundError(f"Bridge checkpoint not found: {bridge_checkpoint}")

    trainer = JointBridgeMPMTrainer(
        cfg,
        args.case_name,
        bridge_checkpoint=bridge_checkpoint,
        inference_dir=args.inference_dir,
        resume_path=physics_checkpoint,
    )
    trainer.train(num_iters=getattr(cfg.bridge, "stage2_iters", 100))

    joint_bridge_output = args.joint_bridge_output or os.path.join(
        cfg.output_dir, args.case_name, "bridge_stage2", "final_bridge_joint.pt"
    )
    trainer.bridge_bundle.save_checkpoint(
        joint_bridge_output,
        extra={
            "stage": "joint",
            "physics_checkpoint": physics_checkpoint,
            "source_bridge_checkpoint": bridge_checkpoint,
        },
    )
    print(f"[JOINT] Saved updated bridge checkpoint to {joint_bridge_output}")


if __name__ == "__main__":
    main()
