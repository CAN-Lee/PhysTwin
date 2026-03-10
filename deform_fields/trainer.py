import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .model import MeshDeformationModel, arap_loss
from .dataset import MeshSequenceDataset
from .utils import compute_se3_clusters, get_mesh_edges
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

class MeshSequenceTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        # Load Dataset
        self.dataset = MeshSequenceDataset(cfg['data_path'], cfg.get('mesh_path'))
        self.can_data = self.dataset.get_canonical_data()
        self.v_canonical = self.can_data['v_canonical'].to(self.device)
        self.faces = self.can_data['faces']
        if self.faces is not None:
            self.faces = self.faces.to(self.device)
            # Precompute edges for ARAP
            self.edges = get_mesh_edges(self.faces)
        else:
            self.edges = None
            
        # Initialize Model with SE(3) Clustering
        self.num_vertices = self.v_canonical.shape[0]
        self.num_clusters = cfg.get('num_clusters', 10)
        self.num_frames = self.dataset.num_frames
        
        # 1. Compute SE(3) Motion Primitives via Clustering and Procrustes alignment
        print(f"Initializing SE(3) clusters with {self.num_clusters} motion primitives...")
        init_rot, init_trans, init_weights = compute_se3_clusters(
            self.dataset.target_pcds, self.num_clusters, device=self.device, v_canonical=self.v_canonical
        )
        
        # 2. Setup Model
        self.model = MeshDeformationModel(
            num_vertices=self.num_vertices,
            num_clusters=self.num_clusters,
            num_frames=self.num_frames,
            device=self.device,
            use_dqs=cfg.get('use_dqs', False),
            v_canonical=self.v_canonical,
            init_rot=init_rot,
            init_trans=init_trans,
            init_weights=init_weights
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.get('lr', 1e-3))
        
        self.log_dir = cfg.get('log_dir', 'deform_fields/logs/mesh_sequence')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self, num_epochs=100):
        print(f"Starting MeshSequence Training...")
        self.model.train()
        
        lambda_temporal = self.cfg.get('lambda_temporal', 0.1)
        lambda_arap = self.cfg.get('lambda_arap', 0.1)
        lambda_rot_reg = self.cfg.get('lambda_rot_reg', 0.01) # Small reg to handle collinearity
        
        pbar = tqdm(range(num_epochs), desc="Epochs")
        for epoch in pbar:
            epoch_loss = 0
            
            # Sequential optimization of frames to maintain temporal smoothness
            # In each epoch, we iterate through frames
            for t in range(self.num_frames):
                self.optimizer.zero_grad()
                
                # Get target point cloud
                target_pcd = self.dataset[t]['target_pcd'].to(self.device).unsqueeze(0)
                
                # Forward: deform v_canonical to current frame t
                vt = self.model(self.v_canonical, t)
                
                # 1. Chamfer Loss
                cham_loss, _ = chamfer_distance(vt.unsqueeze(0), target_pcd)
                
                # 2. ARAP Regularization
                loss_arap = torch.tensor(0.0, device=self.device)
                if self.edges is not None:
                    loss_arap = arap_loss(vt, self.v_canonical, self.edges)
                
                # 3. Temporal Smoothness (on transformations)
                loss_temporal = torch.tensor(0.0, device=self.device)
                if t > 0:
                    loss_temporal += torch.mean((self.model.rotations[t] - self.model.rotations[t-1])**2)
                    loss_temporal += torch.mean((self.model.translations[t] - self.model.translations[t-1])**2)
                
                # 4. Rotation Regularization (to identity)
                # Especially helpful for thin objects like ropes
                loss_rot_reg = self.model.get_rotation_reg_loss()

                loss = cham_loss + lambda_arap * loss_arap + lambda_temporal * loss_temporal + lambda_rot_reg * loss_rot_reg
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()

                # TensorBoard Logging per step
                global_step = epoch * self.num_frames + t
                self.writer.add_scalar('Loss/Total', loss.item(), global_step)
                self.writer.add_scalar('Loss/Chamfer', cham_loss.item(), global_step)
                self.writer.add_scalar('Loss/ARAP', loss_arap.item(), global_step)
                self.writer.add_scalar('Loss/Temporal', loss_temporal.item(), global_step)
                self.writer.add_scalar('Loss/RotReg', loss_rot_reg.item(), global_step)
            
            # TensorBoard Logging per epoch
            self.writer.add_scalar('Loss/Epoch_Avg', epoch_loss / self.num_frames, epoch)
            pbar.set_postfix({'avg_loss': epoch_loss / self.num_frames})

            if (epoch + 1) % self.cfg.get('save_freq', 10) == 0:
                self.save_checkpoint(epoch + 1)
        
        self.writer.close()

    def save_checkpoint(self, epoch):
        path = os.path.join(self.log_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint to {path}")

    def save_results(self, output_path):
        self.model.eval()
        results = []
        with torch.no_grad():
            for t in range(self.num_frames):
                vt = self.model(self.v_canonical, t)
                results.append(vt.cpu().numpy())
        
        output_data = {
            'vertices': np.array(results),
            'v_canonical': self.v_canonical.detach().cpu().numpy(),
            'faces': self.faces.detach().cpu().numpy() if self.faces is not None else None,
            'weights': self.model.get_skinning_weights().detach().cpu().numpy()
        }
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Saved results to {output_path}")
