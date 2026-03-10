import os
import argparse
import pickle
import torch
from omegaconf import OmegaConf
from phys_expert.engine.trainer_mpm import PhysExpertMPMTrainer

def main():
    parser = argparse.ArgumentParser(description="Visualize MPM parameters from a specific iteration")
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_1")
    parser.add_argument("--config", type=str, default="configs/mpm_cloth.yaml")
    parser.add_argument("--params_path", type=str, required=True, help="Path to params_iter_X.pkl")
    parser.add_argument("--output_name", type=str, default="iter_10_visualization.mp4")
    args = parser.parse_args()

    # 1. Load Config
    cfg = OmegaConf.load(args.config)
    cfg.mpm.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Initialize Trainer
    trainer = PhysExpertMPMTrainer(cfg, args.case_name)
    
    # 3. Load Parameters from PKL
    if not os.path.exists(args.params_path):
        print(f"Error: Params file not found at {args.params_path}")
        return
        
    with open(args.params_path, 'rb') as f:
        opt_data = pickle.load(f)
    
    print(f"Loaded parameters from {args.params_path}")
    
    # Inject parameters into trainer's nn.Parameters
    with torch.no_grad():
        # weights -> log_weights
        weights = torch.tensor(opt_data['weights'], device=trainer.device)
        trainer.log_weights.copy_(torch.log(weights + 1e-8))
        
        # mu, lam, fiber_k -> raw_params (inverse sigmoid)
        mu = torch.tensor(opt_data['mu'], device=trainer.device)
        lam = torch.tensor(opt_data['lam'], device=trainer.device)
        fk = torch.tensor(opt_data['fiber_k'], device=trainer.device)
        fdir = torch.tensor(opt_data['fiber_dir'], device=trainer.device)
        
        # We need to map [1e3, 1e5] range back to logit space if clamped, 
        # but here we just do a direct mapping for visualization.
        raw = torch.zeros_like(trainer.raw_params)
        raw[:, 0] = torch.logit(torch.clamp(mu / 1e5, 1e-4, 1-1e-4))
        raw[:, 1] = torch.logit(torch.clamp(lam / 1e5, 1e-4, 1-1e-4))
        raw[:, 2] = torch.logit(torch.clamp(fk / 1e5, 1e-4, 1-1e-4))
        raw[:, 3:6] = fdir 
        
        trainer.raw_params.copy_(raw)

    # 4. Generate Video
    video_path = os.path.join(cfg.output_dir, args.case_name, args.output_name)
    print(f"Starting simulation and rendering...")
    trainer.visualize(video_path)
    print(f"Done! Video saved to: {video_path}")

if __name__ == "__main__":
    main()
