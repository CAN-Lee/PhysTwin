import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class MeshSequenceDataset(Dataset):
    def __init__(self, data_path, mesh_path=None):
        """
        data_path: Path to final_data.pkl
        mesh_path: Path to reference mesh (.obj/.ply). If None, will use first frame PCD as canonical.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
        # object_points: (T, N_pcd, 3)
        self.target_pcds = torch.from_numpy(self.data['object_points']).float()
        self.num_frames = self.target_pcds.shape[0]
        
        # Canonical vertices (Reference Mesh)
        if mesh_path and os.path.exists(mesh_path):
            import trimesh
            mesh = trimesh.load(mesh_path)
            # Handle Scene objects (common for GLB/GLTF)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
                
            self.v_canonical = torch.from_numpy(mesh.vertices).float()
            self.faces = torch.from_numpy(mesh.faces).long()
        else:
            # Fallback: use first frame's point cloud as canonical vertices
            self.v_canonical = self.target_pcds[0]
            self.faces = None # No topology available
            
    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        return {
            'target_pcd': self.target_pcds[idx],
            'frame_idx': idx
        }

    def get_canonical_data(self):
        return {
            'v_canonical': self.v_canonical,
            'faces': self.faces
        }
