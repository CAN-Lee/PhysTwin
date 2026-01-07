"""
Batch training script for third-stage: PhysicsNet training
This script trains PhysicsNet for all cases, loading checkpoints from second-stage training.
"""
import glob
import os
import json

base_path = "./data/different_types"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    # Train PhysicsNet only (freeze physical parameters)
    # To also fine-tune physical parameters, add --train_physics_params flag
    os.system(
        f"python train_physics_net.py "
        f"--base_path {base_path} "
        f"--case_name {case_name} "
        f"--train_frame {train_frame} "
        f"--physics_net_lr 1e-4"  # Lower LR for fine-tuning PhysicsNet
    )

