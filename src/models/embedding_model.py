import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import ConvNextV2Model
from src.models.CBAM import CBAM
from src.models.PWCA import PairwiseCrossAttention


class SubCenterArcFaceHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.k = config.ARCFACE_SUB_CENTERS_K
        self.margin = config.ARCFACE_MARGIN
        self.scale = config.ARCFACE_SCALE
        
        self.weight = nn.Parameter(torch.Tensor(config.NUM_CLASSES * self.k, config.EMBEDDING_DIM))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embedding, labels):
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        kernel_norm = F.normalize(self.weight, p=2, dim=1)
        
        cos_theta = F.linear(embedding_norm, kernel_norm)
        cos_theta = cos_theta.view(-1, self.config.NUM_CLASSES, self.k)

        one_hot_labels = F.one_hot(labels, num_classes=self.config.NUM_CLASSES).view(-1, self.config.NUM_CLASSES, 1)
        positive_sub_centers_cos = cos_theta * one_hot_labels
        hardest_positive_cos, _ = torch.max(positive_sub_centers_cos, dim=2)
        
        cos_theta_yi = torch.gather(hardest_positive_cos, 1, labels.view(-1, 1)).squeeze(1)

        sine = torch.sqrt(1.0 - torch.pow(cos_theta_yi, 2).clamp(0, 1))
        phi = cos_theta_yi * self.cos_m - sine * self.sin_m
        phi = torch.where(cos_theta_yi > self.th, phi, cos_theta_yi - self.mm)

        max_cos_theta_j, _ = torch.max(cos_theta, dim=2)
        
        output_logits = max_cos_theta_j
        output_logits.scatter_(1, labels.view(-1, 1), phi.view(-1, 1))
        
        output_logits *= self.scale
        return output_logits


# class EmbeddingModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         self.backbone = ConvNextV2Model.from_pretrained(config.FBACKBONE)
#         self.feature_dims = self.backbone.config.hidden_sizes
#         self._insert_cbam_modules()

#         feature_dim = self.feature_dims[-1]
#         self.pwca = PairwiseCrossAttention(in_dim=feature_dim)
#         self.global_pool = nn.AdaptiveAvgPool2d(1)

#         self.embedding_head = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim // 2),
#             nn.BatchNorm1d(feature_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim // 2, config.EMBEDDING_DIM)
#         )
        
#         self.arcface_head = SubCenterArcFaceHead(config)

#     def _insert_cbam_modules(self):
#         self.cbam_stages = nn.ModuleList()
#         for dim in self.feature_dims:
#             self.cbam_stages.append(CBAM(dim))

#     def _get_features(self, x):
#         outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        
#         features = outputs.hidden_states[1:]
        
#         refined_features = [self.cbam_stages[i](f) for i, f in enumerate(features)]
#         return refined_features[-1]

#     def forward(self, x, labels=None, x_distractor=None):
#         final_features = self._get_features(x)

#         if self.training and x_distractor is not None:
#             distractor_features = self._get_features(x_distractor)
#             final_features = self.pwca(final_features, distractor_features)

#         pooled_features = self.global_pool(final_features).flatten(1)
#         embedding = self.embedding_head(pooled_features)
        
#         logits = None
#         if labels is not None:
#             logits = self.arcface_head(embedding, labels)
        
#         return {"embedding": embedding, "logits": logits}

#     def freeze_backbone(self):
#         print("Freezing backbone.")
#         for param in self.backbone.parameters():
#             param.requires_grad = False

#     def unfreeze_backbone(self):
#         print("Unfreezing backbone.")
#         for param in self.backbone.parameters():
#             param.requires_grad = True


import timm
class EmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = timm.create_model(config.TBACKBONE, pretrained=True, num_classes=0, features_only=True)
        self._insert_cbam_modules()

        feature_dim = self.backbone.feature_info.info[-1]['num_chs']
        self.pwca = PairwiseCrossAttention(in_dim=feature_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, config.EMBEDDING_DIM)
        )
        
        self.arcface_head = SubCenterArcFaceHead(config)

    def _insert_cbam_modules(self):
        self.cbam_stages = nn.ModuleList()
        for i, stage_info in enumerate(self.backbone.feature_info.info):
            self.cbam_stages.append(CBAM(stage_info['num_chs']))

    def forward(self, x, labels=None, x_distractor=None):
        features = self.backbone(x)
        refined_features = [self.cbam_stages[i](f) for i, f in enumerate(features)]
        final_features = refined_features[-1]

        if self.training and x_distractor is not None:
            distractor_features = self.backbone(x_distractor)[-1]
            distractor_features = self.cbam_stages[-1](distractor_features)
            final_features = self.pwca(final_features, distractor_features)

        pooled_features = self.global_pool(final_features).flatten(1)
        embedding = self.embedding_head(pooled_features)
        
        logits = None
        if labels is not None:
            logits = self.arcface_head(embedding, labels)
        
        return {"embedding": embedding, "logits": logits}

    def freeze_backbone(self):
        print("Freezing backbone.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing backbone.")
        for param in self.backbone.parameters():
            param.requires_grad = True
