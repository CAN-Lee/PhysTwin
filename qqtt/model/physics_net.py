import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather

class GroupEncoder(nn.Module):
    """
    Encodes local geometric and feature information for each group.
    Equivalent to the local PointNet in KNNTransformer.
    """
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim // 2, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # x: (B, C, G, K) - Features of K neighbors for G groups
        B, C, G, K = x.shape
        
        # Reshape for Conv1d: (B * G, C, K)
        x = x.permute(0, 2, 1, 3).reshape(B * G, C, K)
        
        # First layer
        x = F.relu(self.bn1(self.conv1(x))) # (B*G, H/2, K)
        
        # Max pooling over neighbors -> Global feature for the group
        x_global = torch.max(x, dim=2, keepdim=True)[0] # (B*G, H/2, 1)
        
        # Concatenate local and global features
        x_global = x_global.repeat(1, 1, K) # (B*G, H/2, K)
        x = torch.cat([x, x_global], dim=1) # (B*G, H, K)
        
        # Second layer
        x = F.relu(self.bn2(self.conv2(x))) # (B*G, H, K)
        
        # Final max pooling to get group feature
        x = torch.max(x, dim=2)[0] # (B*G, H)
        
        # Reshape back: (B, G, H)
        x = x.reshape(B, G, -1)
        return x

class PhysicsNet(nn.Module):
    def __init__(
        self, 
        in_channels=3+1+3+4, # pos(3) + opacity(1) + scale(3) + rot(4)
        hidden_dim=128, 
        num_experts=3,
        num_groups=512, 
        num_neighbors=32,
        num_transformer_layers=2,
        num_heads=4
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_neighbors = num_neighbors
        self.hidden_dim = hidden_dim
        
        # 1. Group Encoder
        # Input features will be relative_pos (3) + original_features (in_channels)
        self.group_encoder = GroupEncoder(in_channels + 3, hidden_dim)
        
        # 2. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Positional Encoding for Transformer
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. Decoders
        # Expert Weights Decoder
        self.expert_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
    def forward(self, pos, features):
        """
        Args:
            pos: (B, N, 3) - Particle positions
            features: (B, N, C) - Particle features (opacity, scale, etc.)
        Returns:
            expert_weights: (B, N, num_experts) - Softmaxed weights for each particle
        """
        B, N, _ = pos.shape
        
        # ------------------------------------------------------------------
        # 1. Grouping (FPS + KNN)
        # ------------------------------------------------------------------
        
        # FPS to select G center points
        # centers: (B, G, 3), indices: (B, G)
        centers, center_indices = sample_farthest_points(pos, K=self.num_groups)
        
        # KNN to find neighbors for each center
        # nn_dists: (B, G, K), nn_indices: (B, G, K)
        _, nn_indices, _ = knn_points(centers, pos, K=self.num_neighbors)
        
        # Gather features for neighbors
        # (B, G, K, 3)
        group_pos = knn_gather(pos, nn_indices) 
        # (B, G, K, C)
        group_features = knn_gather(features, nn_indices)
        
        # Normalize position: relative to center
        # centers_expanded: (B, G, 1, 3)
        centers_expanded = centers.unsqueeze(2)
        group_pos_rel = group_pos - centers_expanded
        
        # Prepare input for encoder: concat(relative_pos, features)
        # (B, G, K, 3+C) -> Permute to (B, 3+C, G, K) for Conv1d
        group_input = torch.cat([group_pos_rel, group_features], dim=-1)
        group_input = group_input.permute(0, 3, 1, 2)
        
        # ------------------------------------------------------------------
        # 2. Encoding & Transformer
        # ------------------------------------------------------------------
        
        # Encode groups -> (B, G, H)
        group_feats = self.group_encoder(group_input)
        
        # Add positional embedding based on center coordinates
        pos_emb = self.pos_embedding(centers) # (B, G, H)
        group_feats = group_feats + pos_emb
        
        # Transformer processing
        group_feats = self.transformer(group_feats) # (B, G, H)
        
        # ------------------------------------------------------------------
        # 3. Decoding & Interpolation
        # ------------------------------------------------------------------
        
        # Predict weights for group centers: (B, G, num_experts)
        center_weights_logits = self.expert_decoder(group_feats)
        
        # Interpolate back to all N particles
        # Find nearest center for each original particle
        # indices: (B, N, 1) - index of the nearest center for each point
        # dists: (B, N, 1)
        _, indices, _ = knn_points(pos, centers, K=3) # Use 3-NN interpolation for smoothness
        
        # Inverse distance weighting
        # Gather the logits from the 3 nearest centers
        # indices is (B, N, 3)
        # center_weights_logits is (B, G, E)
        # gathered_logits: (B, N, 3, E)
        
        # Pytorch3d's knn_gather assumes (B, N, C), our logits are (B, G, E)
        # We need to act as if G is the source point cloud
        neighbor_logits = knn_gather(center_weights_logits, indices)
        
        # Simple average of 3 nearest centers (or 1-NN if K=1)
        # For better smoothness, we could use IDW, but average is stable
        particle_logits = torch.mean(neighbor_logits, dim=2) # (B, N, E)
        
        # Apply Gumbel Softmax (Hard for forward, Soft for backward)
        # During inference/eval, we might want purely Hard.
        if self.training:
            weights = F.gumbel_softmax(particle_logits, tau=1.0, hard=True, dim=-1)
        else:
            # Standard softmax + argmax logic (or still gumbel with hard=True)
            weights = F.gumbel_softmax(particle_logits, tau=1.0, hard=True, dim=-1)
            
        return weights

