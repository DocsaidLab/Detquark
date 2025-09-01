import copy
import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "auto_pad",
    "make_divisible",
    "ConvBNActivation",
    "DWConv",
]

IntOrSeq = Union[int, Sequence[int]]
PadType = Union[int, Tuple[int, ...]]


# ----------------------------------------------------------------------------- #
# Utility functions
# ----------------------------------------------------------------------------- #
def auto_pad(
    kernel_size: IntOrSeq,
    padding: IntOrSeq | None = None,
    dilation: int = 1,
) -> PadType:
    """Return symmetric padding for “same” spatial output.

    Args:
        kernel_size (int | Sequence[int]): Convolution kernel size.
        padding (int | Sequence[int] | None): User‑supplied padding. If ``None``,
            it is computed automatically.
        dilation (int): Dilation factor applied to the kernel.

    Returns:
        int | tuple[int, ...]: Padding value(s) suitable for ``nn.Conv2d``.
    """
    # Effective kernel under dilation.
    if dilation > 1:
        if isinstance(kernel_size, int):
            kernel_size = dilation * (kernel_size - 1) + 1
        else:
            kernel_size = tuple(dilation * (k - 1) + 1 for k in kernel_size)

    # Auto‑calculate symmetric padding.
    if padding is None:
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(k // 2 for k in kernel_size)

    # ``nn.Conv2d`` expects int or tuple, not list.
    if isinstance(padding, list):
        padding = tuple(padding)

    return padding  # type: ignore[return-value]


def make_divisible(value: int, divisor: int = 8) -> int:
    """Round channel count so it is divisible by ``divisor``.

    This follows the MobileNet/YOLO heuristic, ensuring the new value is
    **not less than 90 %** of the original.

    Args:
        value (int): Original channel count.
        divisor (int): Alignment divisor. Defaults to ``8``.

    Returns:
        int: Rounded channel count.
    """
    new = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new < 0.9 * value:
        new += divisor
    return new


# ----------------------------------------------------------------------------- #
# Building blocks
# ----------------------------------------------------------------------------- #
class ConvBNActivation(nn.Module):
    """Conv2d → BatchNorm2d → optional activation.

    Bias is disabled on the convolution because the subsequent BatchNorm
    contains learnable affine parameters.

    Example:
        >>> layer = ConvBNActivation(16, 32, kernel_size=3, stride=2)
        >>> y = layer(torch.randn(1, 16, 64, 64))

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | Sequence[int], optional): Kernel size. Defaults to ``1``.
        stride (int, optional): Convolution stride. Defaults to ``1``.
        padding (int | Sequence[int] | None, optional): Padding. If ``None``,
            uses :func:`auto_pad`. Defaults to ``None``.
        groups (int, optional): Grouped convolution factor. Defaults to ``1``.
        dilation (int, optional): Dilation rate. Defaults to ``1``.
        activation (bool | nn.Module, optional): Activation selector.

            * ``True`` → new SiLU instance (default)
            * ``False`` → identity (no activation)
            * ``nn.Module`` → deep-copied custom module (independent parameters)
    """

    # A *class‑level* default activation so that external code can override.
    default_act: nn.Module = nn.SiLU()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrSeq = 1,
        stride: int = 1,
        padding: IntOrSeq | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool | nn.Module = True,
    ) -> None:
        super().__init__()

        pad: PadType = auto_pad(kernel_size, padding, dilation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

        # Ensure each block owns a *unique* activation instance.
        if isinstance(activation, nn.Module):
            self.act = copy.deepcopy(activation)
        elif activation:
            self.act = self.default_act.__class__()  # fresh SiLU()
        else:
            self.act = nn.Identity()

    # --------------------------------------------------------------------- #
    # Forward methods
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply conv → bn → act."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv → act (for fused‑BN inference)."""
        return self.act(self.conv(x))


class DWConv(ConvBNActivation):
    """Depth-wise or group-wise convolution block with batch normalization and activation.

    This block performs depth-wise convolution when `in_channels == out_channels`,
    setting `groups` equal to `in_channels`. Otherwise, it falls back to a grouped convolution
    where `groups` is the greatest common divisor (GCD) of `in_channels` and `out_channels`,
    ensuring that both are divisible by `groups`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or Sequence[int], optional): Size of the convolution kernel. Defaults to 1.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Dilation rate of the convolution. Defaults to 1.
        activation (bool or nn.Module, optional): Activation function selector.
            If True, applies SiLU activation (default).
            If False, applies identity (no activation).
            If nn.Module, applies a deep-copied custom activation module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrSeq = 1,
        stride: int = 1,
        dilation: int = 1,
        activation: bool | nn.Module = True,
    ) -> None:
        # Use GCD for grouped conv or depth-wise if equal
        groups: int = math.gcd(in_channels, out_channels)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=None,  # padding auto-calculated in parent class
            groups=groups,
            dilation=dilation,
            activation=activation,
        )
