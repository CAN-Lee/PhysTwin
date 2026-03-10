import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle
from torch.utils.tensorboard import SummaryWriter

from ..model.router_network import PhysicsRouterNetwork
from ..data.dataset_mpm import PhysTwinDataset, collate_fn

class PhysExpertRouterTrainer:
    """
    Stage 2: Distill the optimized physical parameters into the Physics Router Network.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.mpm.device)
        self.global_step = 0
        
        # 1. Logging
        log_dir = os.path.join(cfg.output_dir, 'router_train_logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 2. Data
        self.dataset = PhysTwinDataset(cfg, split='train')
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=cfg.train.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # 3. Model
        self.router = PhysicsRouterNetwork(cfg.router).to(self.device)
        
        # 4. Optimizer
        self.optimizer = optim.Adam(self.router.parameters(), lr=cfg.train.lr_router)
        self.mse_loss = nn.MSELoss()

    def train_epoch(self, epoch):
        self.router.train()
        total_loss = 0.0
        pbar = tqdm(self.dataloader, desc=f"Router Training Epoch {epoch}")
        
        for batch in pbar:
            gaussians = batch['gaussians'].to(self.device)
            tracks = batch['tracks'].to(self.device)
            scene_ids = batch['scene_id']
            B = gaussians.shape[0]
            
            targets_weights = []
            targets_params = []
            valid_mask = []
            
            for scene_id in scene_ids:
                param_path = os.path.join(self.cfg.output_dir, scene_id, "optimized_params.pkl")
                if not os.path.exists(param_path):
                    valid_mask.append(False)
                    continue
                
                with open(param_path, 'rb') as f:
                    opt_data = pickle.load(f)
                
                w = torch.tensor(opt_data['weights'], dtype=torch.float32, device=self.device)
                mu = torch.tensor(opt_data['mu'], dtype=torch.float32, device=self.device)
                lam = torch.tensor(opt_data['lam'], dtype=torch.float32, device=self.device)
                fk = torch.tensor(opt_data['fiber_k'], dtype=torch.float32, device=self.device)
                fdir = torch.tensor(opt_data['fiber_dir'], dtype=torch.float32, device=self.device)
                
                p = torch.zeros(w.shape[0], 8, device=self.device)
                p[:, 0] = mu
                p[:, 1] = lam
                p[:, 2] = fk
                p[:, 3:6] = fdir
                
                targets_weights.append(w)
                targets_params.append(p)
                valid_mask.append(True)
            
            if not any(valid_mask):
                continue
                
            gaussians = gaussians[valid_mask]
            tracks = tracks[valid_mask]
            targets_weights = torch.stack(targets_weights)
            targets_params = torch.stack(targets_params)
            
            pred_weights, pred_params = self.router(gaussians, tracks)
            
            loss_w = self.mse_loss(pred_weights, targets_weights)
            loss_p = self.mse_loss(pred_params[:, :, :6], targets_params[:, :, :6])
            
            loss = loss_w + loss_p * 1e-5
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Router/Total_Loss', loss.item(), self.global_step)
            
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.dataloader)

    def train(self, num_epochs=100):
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} Avg Loss: {avg_loss:.6f}")
            
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(self.cfg.output_dir, f"router_trained_epoch_{epoch+1}.pth")
                torch.save(self.router.state_dict(), ckpt_path)
