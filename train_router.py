import os
import argparse
from omegaconf import OmegaConf
from phys_expert.engine.trainer_router import PhysExpertRouterTrainer

def main():
    parser = argparse.ArgumentParser(description="PhysExpert Stage 2: Router Training")
    parser.add_argument("--config", type=str, default="configs/mpm_cloth.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print("Starting Stage 2: Router Training...")
    trainer = PhysExpertRouterTrainer(cfg)
    trainer.train(num_epochs=args.epochs)
    print("Router Training Completed!")

if __name__ == "__main__":
    main()
