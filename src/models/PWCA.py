import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseCrossAttention(nn.Module):
    """
    Pairwise Cross-Attention (PWCA) module.
    This module is active only during training to regularize feature learning.
    It forces the model to attend to its own unique features while ignoring a distractor.
    """
    def __init__(self, in_dim, num_heads=8):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        
        assert self.head_dim * num_heads == self.in_dim, "in_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(in_dim, in_dim)
        self.k_proj = nn.Linear(in_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)
        self.out_proj = nn.Linear(in_dim, in_dim)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, x_anchor, x_distractor):
        """
        Forward pass for PWCA.
        Args:
            x_anchor (Tensor): Feature map of the anchor image (B, C, H, W)
            x_distractor (Tensor): Feature map of the distractor image (B, C, H, W)
        Returns:
            Tensor: Refined feature map for the anchor image.
        """
        if not self.training:
            # During inference, the module acts as an identity function
            return x_anchor

        B, C, H, W = x_anchor.shape
        
        # Flatten and transpose to (B, N, C) where N = H * W
        x_anchor_flat = x_anchor.flatten(2).transpose(1, 2)
        x_distractor_flat = x_distractor.flatten(2).transpose(1, 2)

        # 1. Project to Q, K, V
        q_anchor = self.q_proj(x_anchor_flat)
        k_anchor = self.k_proj(x_anchor_flat)
        v_anchor = self.v_proj(x_anchor_flat)
        
        k_distractor = self.k_proj(x_distractor_flat)
        v_distractor = self.v_proj(x_distractor_flat)
        
        # 2. Combine Keys and Values from anchor and distractor
        # This forces the anchor query to find relevant info from a combined context
        k_combined = torch.cat([k_anchor, k_distractor], dim=1)
        v_combined = torch.cat([v_anchor, v_distractor], dim=1)

        # 3. Reshape for multi-head attention
        q_anchor = q_anchor.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_combined = k_combined.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_combined = v_combined.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 4. Compute attention scores
        attn_scores = (q_anchor @ k_combined.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 5. Apply attention to combined values
        attn_output = (attn_probs @ v_combined).transpose(1, 2).reshape(B, -1, C)

        # 6. Final projection and reshape back to image format
        refined_features_flat = self.out_proj(attn_output)
        
        # Add residual connection
        refined_features_flat = refined_features_flat + x_anchor_flat
        
        refined_features = refined_features_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return refined_features
