from typing import Union

import torch
import torch.nn as nn

from ..layers import ConvBNActivation
from ..utils import _clone_act

__all__ = ["SpatialPyramidPoolingFastBlock"]


class SpatialPyramidPoolingFastBlock(nn.Module):
    """SPPF - Spatial Pyramid Pooling (fast) layer.

    The block compresses spatial context by applying one 1 x 1 projection,
    followed by **three** successive *k x k* max-pool operations that reuse the
    pooled feature map, and finally a 1 x 1 fusion:

    ::

        y0 = Conv1x1(x)
        y1 = MaxPool_k(y0)
        y2 = MaxPool_k(y1)
        y3 = MaxPool_k(y2)
        out = Conv1x1(cat(y0, y1, y2, y3))

    Notes
    -----
    * Equivalent receptive-field series to SPP(k=(5, 9, 13)) used in YOLOv5.
    * Parameter-free pooling makes this block lightweight and ONNX-friendly.

    Args:
        in_channels (int):  Input feature channels.
        out_channels (int): Output feature channels.
        kernel_size (int, optional): Max-pool size *k*. Defaults to ``5``.
        activation (bool | nn.Module, optional): Activation selector for both
            1 x 1 convolutions.

            * ``True`` → new SiLU instance (default)
            * ``False`` → identity
            * ``nn.Module`` → user-supplied module (deep-copied)
    """

    default_act: nn.Module = nn.SiLU()

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        activation: Union[bool, nn.Module] = True,
    ) -> None:
        super().__init__()

        hidden = in_channels // 2

        act1 = _clone_act(self.default_act, activation)
        self.conv_proj = ConvBNActivation(
            in_channels,
            hidden,
            kernel_size=1,
            stride=1,
            activation=act1,
        )

        act2 = _clone_act(self.default_act, activation)
        self.conv_fuse = ConvBNActivation(
            hidden * 4,
            out_channels,
            kernel_size=1,
            stride=1,
            activation=act2,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x → 1 x 1 proj → pyramid pooling → 1 x 1 fuse."""
        y0 = self.conv_proj(x)
        y1 = self.pool(y0)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv_fuse(torch.cat((y0, y1, y2, y3), dim=1))
