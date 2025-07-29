from typing import List

import torch
import torch.nn as nn
from chameleon import build_neck


class ChameleonNeck(nn.Module):
    """
    Wrapper for a neck module that aggregates backbone features.

    Attributes:
        neck (nn.Module): Instantiated neck network.
    """

    def __init__(self, name: str, **kwargs):
        """
        Build neck module.

        Args:
            name (str): Neck architecture name.
            **kwargs: Additional parameters for neck builder.
        """
        super().__init__()
        self.neck = build_neck(name=name, **kwargs)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate feature maps into a single output tensor.

        Args:
            xs (List[torch.Tensor]): List of feature tensors from backbone.

        Returns:
            torch.Tensor: Aggregated feature representation.
        """
        return self.neck(xs)
