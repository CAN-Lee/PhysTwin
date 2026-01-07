"""
Inference script for PhysicsNet (Third-stage training results)
This script loads the checkpoint from physics_net_experiments and runs inference.
example:
    python inference_physics_net.py \
            --base_path ./data/different_types \
            --case_name weird_package
    xvfb-run -a python inference_physics_net.py \
            --base_path ./data/different_types \
            --case_name weird_package
"""
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to PhysicsNet checkpoint. If None, will look for best model in physics_net_experiments/{case_name}/train/")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    logger.info(f"[DATA TYPE]: {cfg.data_type}")

    # PhysicsNet results are saved in experiments_physics_net
    base_dir = f"experiments_physics_net/{case_name}"

    # Read the first-stage optimized parameters to set the indifferentiable parameters
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="inference_physics_net_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    
    # Find PhysicsNet checkpoint
    if args.checkpoint_path is None:
        train_dir = f"{base_dir}/train"
        if os.path.exists(train_dir):
            # Find all checkpoint files
            checkpoint_files = [f for f in os.listdir(train_dir) if f.startswith("best_") and f.endswith(".pth")]
            if checkpoint_files:
                # Get the epoch number and find the best one
                epochs = [int(f.split("_")[1].split(".")[0]) for f in checkpoint_files]
                best_epoch = max(epochs)
                args.checkpoint_path = f"{train_dir}/best_{best_epoch}.pth"
                logger.info(f"Auto-detected PhysicsNet checkpoint: {args.checkpoint_path}")
            else:
                # Try iter_*.pth files
                iter_files = [f for f in os.listdir(train_dir) if f.startswith("iter_") and f.endswith(".pth")]
                if iter_files:
                    epochs = [int(f.split("_")[1].split(".")[0]) for f in iter_files]
                    latest_epoch = max(epochs)
                    args.checkpoint_path = f"{train_dir}/iter_{latest_epoch}.pth"
                    logger.info(f"Auto-detected PhysicsNet checkpoint: {args.checkpoint_path}")
    
    assert os.path.exists(args.checkpoint_path), f"PhysicsNet checkpoint not found: {args.checkpoint_path}"
    logger.info(f"Loading PhysicsNet checkpoint from: {args.checkpoint_path}")
    
    trainer.test(args.checkpoint_path)
    logger.info("Inference completed!")

