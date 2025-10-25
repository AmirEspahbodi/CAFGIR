# import torch
# import torch.nn as nn
# from .proxy_anchor import ProxyAnchorLoss

# class HybridLoss(nn.Module):
#     """
#     Combines Proxy-Anchor loss (on embeddings) with Label Smoothing Cross-Entropy 
#     loss (on final logits).
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         # Loss for the metric learning part (operates on embeddings)
#         self.proxy_anchor_loss = ProxyAnchorLoss(
#             num_classes=config.NUM_CLASSES,
#             embedding_dim=config.EMBEDDING_DIM,
#             alpha=config.PROXY_ANCHOR_ALPHA,
#             delta=config.PROXY_ANCHOR_DELTA,
#             device=config.DEVICE
#         )
        
#         # Classification loss is now CrossEntropyLoss with Label Smoothing
#         self.classification_loss = nn.CrossEntropyLoss(
#             label_smoothing=config.LABEL_SMOOTHING
#         )
        
#         self.alpha = config.HYBRID_LOSS_ALPHA

#     def forward(self, model_output, labels):
#         # Calculate metric loss on the raw embeddings
#         proxy_loss = self.proxy_anchor_loss(model_output["embedding"], labels)
        
#         # Calculate classification loss on the final, margin-penalized logits
#         # This will now use the label smoothing targets internally
#         arcface_ce_loss = self.classification_loss(model_output["logits"], labels)
        
#         total_loss = proxy_loss + self.alpha * arcface_ce_loss
        
#         return total_loss
