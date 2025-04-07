import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# expects raw logits and target of shape (B, H, W)
cross_entropy_loss = nn.CrossEntropyLoss()


def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.seed(seed)


# model definition
class DeepLabV3Model(nn.Module):
    """DeepLabV3 with a modified classifier head to match the number of classes."""

    def __init__(self, num_classes):
        super().__init__()
        # Load pre-trained DeepLabV3 model
        self.deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
        # Replace the classifier head to match the number of classes
        # The classifier is a Sequential container. Here, index 4 is replaced.
        self.deeplab.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        # reinitialize the new classifier layer
        nn.init.kaiming_normal_(
            self.deeplab.classifier[4].weight, mode="fan_out", nonlinearity="relu"
        )
        if self.deeplab.classifier[4].bias is not None:
            nn.init.constant_(self.deeplab.classifier[4].bias, 0)

    def forward(self, x):
        # The model returns a dict with the output under the key 'out'
        return self.deeplab(x)["out"]
