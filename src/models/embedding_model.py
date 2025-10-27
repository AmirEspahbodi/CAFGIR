import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.PWCA import PairwiseCrossAttention
import timm

class GeM(nn.Module):
    """
    Generalized-Mean (GeM) Pooling layer.
    As proposed in "Fine-tuning CNN Image Retrieval with No Human Annotation"
    (https://arxiv.org/abs/1711.02512)
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: B, C, H, W
        # F.avg_pool2d is used for spatial pooling.
        # We clamp x to avoid nan gradients when x is 0.
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Initialize Backbone
        # We set features_only=True to get intermediate feature maps
        self.backbone = timm.create_model(config.TBACKBONE, pretrained=True, num_classes=0, features_only=True)
        
        # 2. Get feature dimensions for multi-scale concatenation
        # We take the last two feature maps (e.g., stage 3 and stage 4)
        feature_info = self.backbone.feature_info.info
        self.feature_dim_m = feature_info[-2]['num_chs'] # Medium-res features (e.g., from stage 3)
        self.feature_dim_l = feature_info[-1]['num_chs'] # Large-res (coarse) features (e.g., from stage 4)
        
        # 3. Initialize PWCA
        # PWCA will operate on the coarsest (final) feature map
        self.pwca = PairwiseCrossAttention(in_dim=self.feature_dim_l)

        # 4. Initialize GeM Pooling Layers (one for each scale)
        self.global_pool_m = GeM(p=config.GEM_P_INIT)
        self.global_pool_l = GeM(p=config.GEM_P_INIT)

        # 5. Initialize Embedding Head
        # The head now takes the concatenated features as input
        pooled_dim_total = self.feature_dim_m + self.feature_dim_l
        inter_dim = config.EMBEDDING_HEAD_INTERMEDIATE_DIM
        
        self.embedding_head = nn.Sequential(
            nn.Linear(pooled_dim_total, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.ReLU(inplace=True),
            nn.Linear(inter_dim, config.EMBEDDING_DIM)
        )
        

    def forward(self, x, labels=None, x_distractor=None):
        # 1. Get multi-scale features from the backbone
        features = self.backbone(x)
        f_m = features[-2] # Medium-res feature map
        f_l = features[-1] # Large-res (coarse) feature map

        # 2. Apply PWCA (only during training and if distractor is provided)
        if self.training and x_distractor is not None:
            # Get distractor features (only the last map is needed for PWCA)
            distractor_features_l = self.backbone(x_distractor)[-1]
            # Apply PWCA to the coarsest feature map
            f_l = self.pwca(f_l, distractor_features_l)

        # 3. Apply GeM pooling to both feature maps
        pooled_f_m = self.global_pool_m(f_m).flatten(1)
        pooled_f_l = self.global_pool_l(f_l).flatten(1)
        
        # 4. Concatenate pooled features
        pooled_features = torch.cat((pooled_f_m, pooled_f_l), dim=1)

        # 5. Get final embedding
        embedding = self.embedding_head(pooled_features)
        
        logits = None
        return {"embedding": embedding, "logits": logits}

    def freeze_backbone(self):
        print("Freezing backbone.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing backbone.")
        for param in self.backbone.parameters():
            param.requires_grad = True
