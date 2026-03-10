import torch
import os
import argparse
import sys
from omegaconf import OmegaConf

# Add current directory to path so we can import phys_expert
sys.path.append(os.getcwd())
from phys_expert.engine.trainer_mpm import PhysExpertMPMTrainer

def run_test(case_name, config_path, checkpoint_path=None, output_path=None):
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"[Test] Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)
    
    # Ensure output dir exists
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print(f"[Test] Initializing trainer for scene: {case_name}...")
    
    try:
        # Initialize trainer. 
        # If checkpoint_path is provided, we pass None here (auto-load might happen but we overwrite later).
        # Actually, passing None triggers auto-load, which might be slow if it loads a huge checkpoint then we overwrite.
        # But we don't have a "don't load anything" flag in __init__ currently unless we pass a dummy path?
        # It's fine, auto-load is helpful fallback.
        trainer = PhysExpertMPMTrainer(cfg, scene_id=case_name, resume_path=None)
        
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        # import traceback
        # traceback.print_exc()
        return

    if output_path is None:
        if checkpoint_path:
             base = os.path.splitext(os.path.basename(checkpoint_path))[0]
             output_path = os.path.join(cfg.output_dir, case_name, f"test_vis_{base}.mp4")
        else:
             output_path = os.path.join(cfg.output_dir, case_name, "test_vis_auto.mp4")
    
    # Ensure output directory for video exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[Test] Starting test run -> {output_path}")
    
    if checkpoint_path:
        # Use the explicit test method we added
        trainer.test(checkpoint_path, output_path)
    else:
        # Use whatever was loaded during init
        print("[Test] No explicit checkpoint provided. Using state loaded during initialization.")
        trainer.visualize(output_path)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPM Test/Inference with Checkpoint")
    parser.add_argument("--case", type=str, required=True, help="Scene ID (e.g. double_lift_cloth_1)")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt or .pkl checkpoint. If not provided, uses latest available.")
    parser.add_argument("--out", type=str, default=None, help="Output video path")
    
    args = parser.parse_args()
    run_test(args.case, args.config, args.checkpoint, args.out)
