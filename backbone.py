import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class BackboneModel(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = 'vit_base_patch16_224'):
        super().__init__()
        
        self.backbone_type = backbone

        if backbone == 'vit_base_patch16_224':
            self.backbone = ViTModel.from_pretrained(
                'google/vit-base-patch16-224', 
                add_pooling_layer=False
            )
            self.feature_dim = self.backbone.config.hidden_size
        elif backbone == 'efficientnet_v2_s':
            # Load pretrained EfficientNetV2-S
            weights = EfficientNet_V2_S_Weights.DEFAULT
            self.backbone = efficientnet_v2_s(weights=weights)
            
            # Remove the classifier head from the backbone as we will use our own
            # We only need the feature extractor part
            # EfficientNet V2 structure: features -> avgpool -> classifier
            # We will use features + avgpool
            self.feature_dim = self.backbone.classifier[1].in_features
            # We keep the backbone as the full model but will bypass its classifier in get_features
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.get_features(x)
        return self.classifier(features)

    def get_features(self, x):
        if self.backbone_type == 'vit_base_patch16_224':
            # Transformer-based models (ViT)
            outputs = self.backbone(pixel_values=x)
            
            # Check if pooler_output exists and is not None
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                # Use CLS token (first token) from last_hidden_state
                return outputs.last_hidden_state[:, 0]
        elif self.backbone_type == 'efficientnet_v2_s':
            # EfficientNet V2
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
