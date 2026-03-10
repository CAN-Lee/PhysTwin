import os
import argparse
from omegaconf import OmegaConf
from phys_expert.engine.trainer_mpm import PhysExpertMPMTrainer

def main():
    parser = argparse.ArgumentParser(description="PhysExpert Stage 1: MPM Parameter Training")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/mpm_cloth.yaml")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    cfg.mpm.device = 'cuda'
    
    trainer = PhysExpertMPMTrainer(cfg, args.case_name)
    trainer.train(num_iters=args.iters)

if __name__ == "__main__":
    main()
