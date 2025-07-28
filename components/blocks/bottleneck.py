from typing import Sequence

import torch
import torch.nn as nn

from ..layers import ConvBNActivation

__all__ = [
    'BottleneckBlock',
]


class BottleneckBlock(nn.Module):
    """Bottleneck block consisting of an expansion 1 x 1 conv, a projection 3 x 3 conv, and optional shortcut.

    This block first expands the channel dimension by `expansion` via a 1 x 1 convolution,
    then projects back to `out_channels` via a 3 x 3 convolution. If `use_shortcut` is True and
    `in_channels == out_channels`, the input is added to the output (residual).

    Args:
        in_channels: Number of channels in the input tensor.
        out_channels: Number of channels produced by the block.
        use_shortcut: Whether to apply a residual connection (default: True).
        groups: Number of convolution groups for the second conv (default: 1).
        kernel_sizes: Sequence of two ints specifying kernel sizes for conv1 and conv2 (default: (1, 3)).
        expansion: Expansion ratio for the hidden channels (default: 0.5).

    Attributes:
        conv1: 1 x 1 convolution for channel expansion.
        conv2: 3 x 3 convolution for projection.
        can_residual: Whether the residual connection is active.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        kernel_sizes: Sequence[int] = (3, 3),
        expansion: float = 0.5
    ) -> None:
        super().__init__()
        # Compute intermediate channels
        hidden_channels = int(out_channels * expansion)

        # 1 x 1 expansion conv → BN → SiLU
        self.conv1 = ConvBNActivation(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_sizes[0],
            stride=1
        )

        # 3 x 3 projection conv → BN → SiLU
        self.conv2 = ConvBNActivation(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[1],
            stride=1,
            groups=groups
        )

        # Residual is only valid when shapes match
        self.can_residual = shortcut and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bottleneck block to the input tensor.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width).

        Returns:
            Output tensor of shape (batch, out_channels, height, width).
            If residual is active, this is x + conv2(conv1(x)), otherwise just conv2(conv1(x)).
        """
        y = self.conv2(self.conv1(x))
        if self.can_residual:
            return x + y
        return y
