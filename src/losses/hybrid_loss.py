# FILE: hybrid_loss.py

import torch
import torch.nn as nn
from .proxy_anchor import ProxyAnchorLoss

class HybridLoss(nn.Module):
    """
    MODIFIED: This loss now *only* applies Proxy-Anchor loss.
    The ArcFace head and its CE loss are removed to prevent conflicts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Loss for the metric learning part (operates on embeddings)
        self.proxy_anchor_loss = ProxyAnchorLoss(
            num_classes=config.NUM_CLASSES,
            embedding_dim=config.EMBEDDING_DIM,
            alpha=config.PROXY_ANCHOR_ALPHA,
            delta=config.PROXY_ANCHOR_DELTA,
            device=config.DEVICE
        )
        
        # --- REMOVED ---
        # Classification loss is removed
        # self.classification_loss = nn.CrossEntropyLoss(
        #     label_smoothing=config.LABEL_SMOOTHING
        # )
        
        # Alpha is removed
        # self.alpha = config.HYBRID_LOSS_ALPHA
        # --- END REMOVED ---

    def forward(self, model_output, labels):
        # Calculate metric loss on the raw embeddings
        proxy_loss = self.proxy_anchor_loss(model_output["embedding"], labels)
        
        # --- REMOVED ---
        # Classification loss calculation is removed
        # arcface_ce_loss = self.classification_loss(model_output["logits"], labels)
        # 
        # total_loss = proxy_loss + self.alpha * arcface_ce_loss
        # --- END REMOVED ---
        
        return proxy_loss # <-- Return ONLY the proxy loss