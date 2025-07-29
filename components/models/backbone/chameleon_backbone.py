from typing import List

import torch
import torch.nn as nn
from chameleon import build_backbone


class ChameleonBackbone(nn.Module):
    """
    Wrapper for a vision backbone that also records output channel dimensions.

    Attributes:
        backbone (nn.Module): Instantiated backbone network.
        channels (List[int]): Number of output channels at each backbone stage.
    """

    def __init__(self, name: str, **kwargs):
        """
        Build backbone and infer feature map channel dimensions.

        Args:
            name (str): Backbone architecture name.
            **kwargs: Additional parameters for backbone builder.
        """
        super().__init__()
        self.backbone = build_backbone(name=name, **kwargs)

        # Determine output channel sizes by running a dummy input
        with torch.no_grad():
            dummy = torch.rand(1, 3, 128, 128)
            # Collect channel dimension from each feature map
            self.channels = [feat.size(1) for feat in self.backbone(dummy)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward input through the backbone.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch, 3, H, W).

        Returns:
            List[torch.Tensor]: Feature map tensors from backbone stages.
        """
        return self.backbone(x)
