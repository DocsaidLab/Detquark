from typing import List

import timm
import torch
import torch.nn as nn


class TimmBackbone(nn.Module):
    """
    Wrapper for a vision backbone that also records output channel dimensions.

    Attributes:
        backbone (nn.Module): Instantiated backbone network.
        channels (List[int]): Number of output channels at each backbone stage.
    """

    def __init__(
        self,
        name: str,
        features_only: bool = True,
        pretrained: bool = False,
        **kwargs
    ):
        super().__init__()
        self.backbone = timm.create_model(
            name,
            features_only=features_only,
            pretrained=pretrained,
            **kwargs
        )
        self.feature_info = self.backbone.feature_info
        self.channels = [
            feat["num_chs"]
            for feat in self.feature_info
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)
