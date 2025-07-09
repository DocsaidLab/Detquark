from typing import List, Optional

import torch
import torch.nn as nn
from chameleon import build_neck


class YOLOv3Neck(nn.Module):
    """
    Multi-scale feature aggregation module for YOLOv3.

    This class selects a subset of backbone feature maps based on
    `in_indices` and passes them through a neck module (e.g., FPN).

    Attributes:
        in_indices (List[int]): Indices of backbone features to aggregate.
        in_channels_list (List[int]): Channel dimensions of selected features.
        neck (nn.Module): Fusion module built by `build_neck`.
    """

    def __init__(
        self,
        name: str = 'FPN',
        out_channels: int = 512,
        in_indices: Optional[List[int]] = None,
        in_channels_list: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Initialize the YOLOv3Neck.

        Args:
            name (str): Neck architecture identifier passed to `build_neck`.
            out_channels (int): Number of channels for fused output.
            in_indices (List[int], optional): Indices of backbone feature maps to select.
                                                Defaults to None to use all provided features.
            in_channels_list (List[int], optional): Channel sizes of all backbone features.
                                                    Required if `in_indices` is specified.
            **kwargs: Additional arguments forwarded to `build_neck`.

        Raises:
            ValueError: If `in_indices` refers to invalid positions in `in_channels_list`.
        """
        super().__init__()

        # Determine default indices if none provided
        if in_indices is None:
            # Use all features
            in_indices = list(range(len(in_channels_list or [])))

        # Validate against available channel list
        if in_channels_list is None or max(in_indices) >= len(in_channels_list):
            raise ValueError(
                f"Expected at least {max(in_indices)+1} channels in in_channels_list, "
                f"got {len(in_channels_list or [])}"
            )

        self.in_indices = in_indices
        # Filter channel sizes to selected indices
        self.in_channels_list = [in_channels_list[i] for i in in_indices]

        # Build the fusion neck with selected channels
        self.neck = build_neck(
            name=name,
            out_channels=out_channels,
            in_channels_list=self.in_channels_list,
            **kwargs
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for multi-scale fusion.

        Args:
            xs (List[Tensor]): Feature maps from backbone in sequential order.

        Returns:
            Tensor: Single fused feature map from the neck.

        Raises:
            ValueError: If `xs` has fewer elements than required by `in_indices`.
        """
        # Ensure input list covers required indices
        if max(self.in_indices) >= len(xs):
            raise ValueError(
                f"Expected at least {max(self.in_indices)+1} feature maps, got {len(xs)}"
            )

        # Select and fuse features
        selected = [xs[i] for i in self.in_indices]
        return self.neck(selected)
