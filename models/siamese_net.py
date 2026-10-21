import torch
import torch.nn as nn
from torchvision.models import resnet50

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super(SiameseNetwork, self).__init__()
        # Load ResNet-50 backbone
        if pretrained:
            try:
                self.encoder = resnet50(weights='DEFAULT')
            except Exception:
                self.encoder = resnet50(pretrained=True)
        else:
            try:
                self.encoder = resnet50(weights=None)
            except Exception:
                self.encoder = resnet50(pretrained=False)
        
        # Remove the fully connected layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # Classification head for downstream task
        self.classifier = nn.Linear(2048, 5)  # 5 classes
        
        self.embedding_dim = embedding_dim
        
    def forward_once(self, x):
        # Forward pass through encoder
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        return features
    
    def forward(self, x1, x2=None):
        # If two inputs, return embeddings for contrastive loss
        if x2 is not None:
            emb1 = self.forward_once(x1)
            emb2 = self.forward_once(x2)
            proj1 = self.projection_head(emb1)
            proj2 = self.projection_head(emb2)
            return proj1, proj2
        # If single input, return logits for classification
        else:
            features = self.forward_once(x1)
            logits = self.classifier(features)
            return logits
