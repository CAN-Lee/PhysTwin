import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (
    quaternion_to_matrix, 
    matrix_to_quaternion, 
    quaternion_multiply,
    quaternion_invert
)

class MeshDeformationModel(nn.Module):
    def __init__(self, num_vertices, num_clusters, num_frames, device='cuda', use_dqs=False, v_canonical=None, 
                 init_rot=None, init_trans=None, init_weights=None):
        super().__init__()
        self.num_vertices = num_vertices
        self.num_clusters = num_clusters
        self.num_frames = num_frames
        self.device = device
        self.use_dqs = use_dqs

        # Skinning Weights: (N, K)
        if init_weights is not None:
            # init_weights: (N, K) in probability space
            self.skinning_weights = nn.Parameter(torch.log(init_weights.to(device) + 1e-6))
        elif v_canonical is not None and num_clusters > 1:
            from pytorch3d.ops import sample_farthest_points, knn_points
            # Initialize cluster centers using FPS
            # v_canonical: (N, 3) -> (1, N, 3)
            v_can_batch = v_canonical.unsqueeze(0).to(device)
            centers, _ = sample_farthest_points(v_can_batch, K=num_clusters) # (1, K, 3)
            
            # Initial weights based on inverse distance to centers
            # dist: (1, N, K)
            dist, _, _ = knn_points(v_can_batch, centers, K=num_clusters)
            # weights = exp(-dist)
            init_weights = torch.exp(-dist.squeeze(0) * 10.0) # (N, K)
            # Convert to log space for softmax later
            self.skinning_weights = nn.Parameter(torch.log(init_weights + 1e-6))
        else:
            self.skinning_weights = nn.Parameter(torch.ones(num_vertices, num_clusters, device=device) / num_clusters)

        # SE(3) Trajectories: (T, K, 4) for quaternion, (T, K, 3) for translation
        # rotations: (T, K, 4) as (w, x, y, z), translations: (T, K, 3)
        if init_rot is not None:
            self.rotations = nn.Parameter(init_rot.to(device).float())
        else:
            self.rotations = nn.Parameter(torch.zeros(num_frames, num_clusters, 4, device=device))
            self.rotations.data[..., 0] = 1.0 # identity quaternion (w=1)
            
        if init_trans is not None:
            self.translations = nn.Parameter(init_trans.to(device).float())
        else:
            self.translations = nn.Parameter(torch.zeros(num_frames, num_clusters, 3, device=device))

    def get_skinning_weights(self):
        return F.softmax(self.skinning_weights, dim=-1)

    def get_rotation_reg_loss(self):
        """
        Regularization to keep rotations close to identity if not driven by data.
        Helps with collinearity issues where rotation around one axis is unobservable.
        """
        # Penalize (1-w)^2 + x^2 + y^2 + z^2
        identity_quat = torch.zeros_like(self.rotations)
        identity_quat[..., 0] = 1.0
        return torch.mean((self.rotations - identity_quat)**2)

    def forward(self, v_canonical, t_idx):
        """
        v_canonical: (N, 3) - canonical vertex positions
        t_idx: frame index (integer)
        """
        weights = self.get_skinning_weights() # (N, K)
        
        # Get SE(3) for current frame
        quat = self.rotations[t_idx] # (K, 4)
        # Ensure unit quaternions
        quat = F.normalize(quat, dim=-1)
        
        trans = self.translations[t_idx] # (K, 3)

        if self.use_dqs:
            return self.apply_dqs(v_canonical, weights, quat, trans)
        else:
            return self.apply_lbs(v_canonical, weights, quat, trans)

    def apply_lbs(self, v_canonical, weights, quat, trans):
        """
        Linear Blend Skinning
        """
        rot_mats = quaternion_to_matrix(quat) # (K, 3, 3)
        
        # Transform v_canonical by each cluster's transformation
        # Result shape: (N, K, 3)
        # Using einsum for clearer batch matrix multiplication: (N, 3) @ (K, 3, 3)^T -> (N, K, 3)
        # v_transformed[n, k, j] = sum_i (v_canonical[n, i] * rot_mats[k, j, i]) + trans[k, j]
        v_transformed = torch.einsum('ni,kij->nkj', v_canonical, rot_mats) + trans.unsqueeze(0)
        
        # Blend: vt = sum_k w_i,k * v_transformed_k
        # (N, 3)
        vt = torch.sum(weights.unsqueeze(-1) * v_transformed, dim=1)
        return vt

    def apply_dqs(self, v_canonical, weights, quat, trans):
        """
        Dual Quaternion Skinning (Simplified version)
        Note: Proper DQS involves dual quaternions. 
        Here we implement a standard approximation.
        """
        # For now, let's keep it simple with LBS or implement proper DQS if needed.
        # Given the "Volume loss" mention, DQS is better for large rotations.
        # For initial implementation, LBS is often sufficient.
        # TODO: Implement full DQS if required.
        return self.apply_lbs(v_canonical, weights, quat, trans)

def arap_loss(v_deformed, v_original, edges):
    """
    Placeholder for As-Rigid-As-Possible loss.
    """
    # This usually requires local rotation estimation for each vertex/ring.
    # For a simple version, we can penalize edge length changes.
    v_orig_edges = v_original[edges[:, 0]] - v_original[edges[:, 1]]
    v_def_edges = v_deformed[edges[:, 0]] - v_deformed[edges[:, 1]]
    
    orig_lens = torch.norm(v_orig_edges, dim=-1)
    def_lens = torch.norm(v_def_edges, dim=-1)
    
    loss = F.mse_loss(def_lens, orig_lens)
    return loss
