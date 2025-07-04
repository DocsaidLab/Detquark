from typing import List

import torch
import torch.nn as nn
from chameleon import build_backbone, build_neck


class Permute(nn.Module):
    """
    Reorders the dimensions of a tensor.

    Attributes:
        dims (List[int]): Desired ordering of dimensions.
    """

    def __init__(self, dims: List[int]) -> None:
        """
        Initialize Permute module.

        Args:
            dims (List[int]): Sequence of dimension indices to permute.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Permute the input tensor according to `dims`.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Tensor with dimensions reordered.
        """
        return x.permute(*self.dims)


class Transpose(nn.Module):
    """
    Swaps two dimensions of a tensor.

    Attributes:
        dim1 (int): First dimension index.
        dim2 (int): Second dimension index.
    """

    def __init__(self, dim1: int, dim2: int) -> None:
        """
        Initialize Transpose module.

        Args:
            dim1 (int): First dimension to swap.
            dim2 (int): Second dimension to swap.
        """
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose the two specified dimensions of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with `dim1` and `dim2` swapped.
        """
        return x.transpose(self.dim1, self.dim2)


class Backbone(nn.Module):
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


class Neck(nn.Module):
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
