import torch
import os
import sys
import shutil
import pickle
import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phys_expert.engine.trainer_mpm import PhysExpertMPMTrainer

def create_dummy_data(root_dir, max_frames=5, num_particles=100):
    os.makedirs(root_dir, exist_ok=True)
    scene_dir = os.path.join(root_dir, "scene_test_001")
    os.makedirs(scene_dir, exist_ok=True)
    
    # Create final_data.pkl compatible with new Dataset
    # N_surf = num_particles, N_inner = 20
    
    data = {
        "object_points": np.random.rand(max_frames, num_particles, 3).astype(np.float32), # Surface Tracks
        "object_colors": np.random.rand(max_frames, num_particles, 3).astype(np.float32),
        "interior_points": np.random.rand(20, 3).astype(np.float32), # Inner points
        "surface_points": np.zeros((0, 3)).astype(np.float32), # Optional
        "controller_points": np.random.rand(max_frames, 2, 3).astype(np.float32) # 2 control points
    }
    
    with open(os.path.join(scene_dir, "final_data.pkl"), "wb") as f:
        pickle.dump(data, f)
        
    print(f"Created dummy data at {scene_dir}")

def test_training_pipeline():
    print("Testing Training Pipeline...")
    
    # Load Config
    cfg_path = os.path.join(os.path.dirname(__file__), "phys_expert_jelly.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    # Modify Config for fast test
    cfg.data.root = "./output/test_data_pipeline"
    cfg.train.batch_size = 1
    cfg.train.n_epochs = 1
    cfg.mpm.max_frames = 5 # Short sequence
    cfg.mpm.steps_per_frame = 2 # Few steps
    cfg.router.dynamic.n_tracks = 50
    cfg.router.static.n_patches = 8
    cfg.mpm.n_particles = 100
    
    # Update test_router.py to match 6-dim inputs if needed (not here, but test_router.py manually creates inputs)
    
    # Create Dummy Data first
    create_dummy_data(cfg.data.root, max_frames=cfg.mpm.max_frames, num_particles=cfg.mpm.n_particles)
    
    try:
        # Create Trainer
        trainer = PhysExpertMPMTrainer(cfg, "scene_test_001")
        
        # Run 1 epoch
        trainer.train(num_iters=1)
        print("Training Loop Completed Successfully!")
    except Exception as e:
        print(f"Training Loop Failed: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Cleanup
        if os.path.exists(cfg.data.root):
            shutil.rmtree(cfg.data.root)

if __name__ == "__main__":
    test_training_pipeline()
