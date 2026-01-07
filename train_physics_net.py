"""
Third-stage training script: Train PhysicsNet (Neural Constitutive Law)
This script loads the checkpoint from second-stage training and continues training PhysicsNet.
example:
    python train_physics_net.py \
            --base_path ./data/different_types \
            --case_name weird_package \
"""
import warnings
import sys
import os
import logging

# Suppress Warp warnings about set_control_points (control points don't need gradients, this is expected)
# These warnings are harmless but spam the console
# Method 1: Python warnings filter
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*enable_backward=False.*")
warnings.filterwarnings("ignore", message=".*set_control_points.*")
warnings.filterwarnings("ignore", message=".*Running the tape backwards.*")
warnings.filterwarnings("ignore", message=".*may produce incorrect gradients.*")

# Method 2: Suppress warnings from warp module specifically
class WarpWarningFilter(logging.Filter):
    def filter(self, record):
        return "enable_backward=False" not in record.getMessage() and \
               "set_control_points" not in record.getMessage() and \
               "Running the tape backwards" not in record.getMessage()

# Apply filter to root logger (Warp might use it)
logging.getLogger().addFilter(WarpWarningFilter())

from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
import warp as wp
from argparse import ArgumentParser
import os
import pickle
import json


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, default=None,
                        help="Training frame number. If None, will read from split.json")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to second-stage checkpoint. If None, will look for best model in experiments/{case_name}/train/")
    parser.add_argument("--train_physics_params", action="store_true",
                        help="Whether to also fine-tune physical parameters (spring_Y, etc.) along with PhysicsNet")
    parser.add_argument("--physics_net_lr", type=float, default=None,
                        help="Learning rate for PhysicsNet. If None, uses cfg.base_lr")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    
    # Read train_frame from split.json if not provided
    if args.train_frame is None:
        split_path = f"{base_path}/{case_name}/split.json"
        if os.path.exists(split_path):
            with open(split_path, "r") as f:
                split = json.load(f)
            train_frame = split["train"][1]
            logger.info(f"Read train_frame={train_frame} from {split_path}")
        else:
            raise ValueError(f"train_frame not provided and split.json not found at {split_path}")
    else:
        train_frame = args.train_frame

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    print(f"[DATA TYPE]: {cfg.data_type}")

    # Use a completely separate directory for third-stage training
    # This keeps physics_net results isolated from second-stage results
    base_dir = f"experiments_physics_net/{case_name}"
    second_stage_dir = f"experiments/{case_name}"  # Directory for loading second-stage checkpoint

    # Read the first-stage optimized parameters
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
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

    logger.set_log_file(path=base_dir, name="physics_net_log")
    
    # Initialize trainer
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
    )

    # Load second-stage checkpoint
    if args.checkpoint_path is None:
        # Look for best model in the second-stage training directory
        train_dir = f"{second_stage_dir}/train"
        if os.path.exists(train_dir):
            # Find all checkpoint files
            checkpoint_files = [f for f in os.listdir(train_dir) if f.startswith("best_") and f.endswith(".pth")]
            if checkpoint_files:
                # Extract epoch numbers and find the latest
                epochs = [int(f.split("_")[1].split(".")[0]) for f in checkpoint_files]
                best_epoch = max(epochs)
                args.checkpoint_path = f"{train_dir}/best_{best_epoch}.pth"
                logger.info(f"Auto-detected checkpoint: {args.checkpoint_path}")
            else:
                # Fallback to latest iter_ file
                iter_files = [f for f in os.listdir(train_dir) if f.startswith("iter_") and f.endswith(".pth")]
                if iter_files:
                    epochs = [int(f.split("_")[1].split(".")[0]) for f in iter_files]
                    latest_epoch = max(epochs)
                    args.checkpoint_path = f"{train_dir}/iter_{latest_epoch}.pth"
                    logger.info(f"Auto-detected checkpoint: {args.checkpoint_path}")

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=cfg.device)

        # Load physical parameters
        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]

        assert (
            len(spring_Y) == trainer.simulator.n_springs
        ), "Checkpoint spring_Y size doesn't match simulator"

        trainer.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        trainer.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        trainer.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        # Load PhysicsNet if available
        if "physics_net" in checkpoint:
            trainer.physics_net.load_state_dict(checkpoint["physics_net"])
            logger.info("Loaded PhysicsNet from checkpoint")
        else:
            logger.info("No PhysicsNet found in checkpoint, starting with random initialization")

        # Load optimizer state if available
        if "optimizer_state_dict" in checkpoint:
            # Rebuild optimizer with correct parameters
            physics_net_lr = args.physics_net_lr if args.physics_net_lr is not None else cfg.base_lr
            
            if args.train_physics_params:
                # Train both PhysicsNet and physical parameters
                trainer.optimizer = torch.optim.Adam(
                    [
                        {"params": trainer.physics_net.parameters(), "lr": physics_net_lr},
                        {"params": [wp.to_torch(trainer.simulator.wp_spring_Y)], "lr": cfg.base_lr},
                        {"params": [wp.to_torch(trainer.simulator.wp_collide_elas)], "lr": cfg.base_lr},
                        {"params": [wp.to_torch(trainer.simulator.wp_collide_fric)], "lr": cfg.base_lr},
                        {"params": [wp.to_torch(trainer.simulator.wp_collide_object_elas)], "lr": cfg.base_lr},
                        {"params": [wp.to_torch(trainer.simulator.wp_collide_object_fric)], "lr": cfg.base_lr},
                    ],
                    lr=cfg.base_lr,
                    betas=(0.9, 0.99),
                )
            else:
                # Only train PhysicsNet, freeze physical parameters
                trainer.simulator.wp_spring_Y.requires_grad = False
                trainer.simulator.wp_collide_elas.requires_grad = False
                trainer.simulator.wp_collide_fric.requires_grad = False
                trainer.simulator.wp_collide_object_elas.requires_grad = False
                trainer.simulator.wp_collide_object_fric.requires_grad = False
                
                trainer.optimizer = torch.optim.Adam(
                    [{"params": trainer.physics_net.parameters(), "lr": physics_net_lr}],
                    lr=physics_net_lr,
                    betas=(0.9, 0.99),
                )
            
            # Try to load optimizer state (may fail if structure changed, that's ok)
            try:
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state from checkpoint")
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}. Starting with fresh optimizer.")
        
        logger.info(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        logger.warning(f"Checkpoint not found at {args.checkpoint_path}. Starting training from scratch.")
        if args.checkpoint_path:
            logger.warning("This will initialize PhysicsNet randomly. Make sure this is intended!")

    # Start training
    logger.info("=" * 60)
    logger.info("Starting Third-Stage Training: PhysicsNet (Neural Constitutive Law)")
    logger.info(f"Training PhysicsNet only: {not args.train_physics_params}")
    logger.info(f"PhysicsNet LR: {args.physics_net_lr if args.physics_net_lr else cfg.base_lr}")
    logger.info("=" * 60)
    
    trainer.train(start_epoch=-1)  # Start from beginning of training loop

