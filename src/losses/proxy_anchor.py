import torch
import torch.nn as nn
import torch.nn.functional as F

class ProxyAnchorLoss(nn.Module):
    """Proxy-Anchor Loss"""
    def __init__(self, num_classes, embedding_dim, alpha=32.0, delta=0.1, device='cuda'):
        super(ProxyAnchorLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.delta = delta
        
        # Proxies are learnable parameters
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim).to(device))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, embeddings, labels):
        # Normalize both embeddings and proxies
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        proxies_norm = F.normalize(self.proxies, p=2, dim=1)

        # Calculate cosine similarity
        cos_sim = F.linear(embeddings_norm, proxies_norm)

        # Create masks for positive and negative proxies
        pos_mask = F.one_hot(labels, self.num_classes).float()
        neg_mask = 1 - pos_mask

        # Hardest positive: The positive proxy with the smallest similarity
        with torch.no_grad():
            pos_sim = cos_sim.clone().masked_fill_(neg_mask.bool(), -1e9)
            hardest_pos_sim, _ = pos_sim.max(dim=1, keepdim=True)
        
        # Hardest negative: The negative proxy with the largest similarity
        with torch.no_grad():
            neg_sim = cos_sim.clone().masked_fill_(pos_mask.bool(), 1e9)
            hardest_neg_sim, _ = neg_sim.min(dim=1, keepdim=True)
            
        # Log-Sum-Exp for positive part
        pos_term = torch.log(1 + torch.sum(torch.exp(-self.alpha * (cos_sim - self.delta)) * pos_mask, dim=1))
        
        # Log-Sum-Exp for negative part
        neg_term = torch.log(1 + torch.sum(torch.exp(self.alpha * (cos_sim + self.delta)) * neg_mask, dim=1))
        
        loss = (torch.mean(pos_term) + torch.mean(neg_term))
        return loss
