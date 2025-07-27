from typing import Union

import torch
import torch.nn as nn

from ..layers import PositionSensitiveAttention
from ..layers.common import ConvBNActivation
from ..utils import _clone_act
from .bottleneck import BottleneckBlock

__all__ = [
    'CSPPointwiseResidualBlock',
    'CSPDualConvBottleneckBlock',
    'CSPDualConvFastBottleneckBlock',
    'CSPTripleConvBottleneckBlock',
    'CSPTripleConvKernelBlock',
    'CSPKernelMixFastBottleneckBlock',
    'CSPDualPSAStackBlock',
]


class CSPPointwiseResidualBlock(nn.Module):
    """CSP bottleneck with point-wise projection and residual 3x3 conv stack.

    Block structure::

        y  = Conv1x1(x)                      # point-wise projection
        z  = Sequential([Conv3x3] * n)(y)    # residual branch
        out = z + y

    Args:
        in_channels (int):  Number of input feature channels.
        out_channels (int): Number of output feature channels.
        num_blocks (int, optional): How many 3x3 conv layers to stack in the
            residual branch. Defaults to ``1``.

    Example:
        >>> blk = CSPPointwiseResidualBlock(64, 128, num_blocks=2)
        >>> y = blk(torch.randn(1, 64, 56, 56))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()

        # 1 x 1 projection (no activation in residual addition style)
        self.pw_conv = ConvBNActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            activation=True,  # 可以改成 False 以貼近部分原論文實作
        )

        # Residual 3 x 3 conv stack
        self.res_stack = nn.Sequential(
            *(
                ConvBNActivation(out_channels, out_channels, kernel_size=3)
                for _ in range(num_blocks)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x → point-wise conv → residual stack → add."""
        y = self.pw_conv(x)
        return self.res_stack(y) + y


class CSPDualConvBottleneckBlock(nn.Module):
    """CSP bottleneck with dual point-wise convolutions and an inner Bottleneck stack.

    Block diagram::

        ┌───────────────┐
        │     x         │
        └───────────────┘
               │
        ┌──────▼───── Conv1x1 (2·h) ───────┐
        │                                  │
      Split                             Split
        │                                  │
   ┌────▼────┐                       ┌─────▼────┐
   │   y1    │                       │    y2    │
   └─────────┘                       └──────────┘
        │                                   │
   Bottleneck x n                           │
        │                                   │
        └───────► Concat(y1, y2) ◄──────────┘
                      │
                 Conv1x1 (out)
                      │
                    out

    Args:
        in_channels (int):   Number of input feature channels.
        out_channels (int):  Number of output feature channels.
        num_blocks (int, optional): Number of inner :class:`Bottleneck`
            layers. Defaults to ``1``.
        shortcut (bool, optional): Enable shortcut inside each Bottleneck.
            Defaults to ``True``.
        groups (int, optional): Convolution groups for Bottleneck layers.
            Defaults to ``1``.
        expansion (float, optional): Hidden-channel expansion ratio.
            Defaults to ``0.5``.
        activation (bool | nn.Module, optional): Activation selector.
            * ``True`` → fresh SiLU (default)
            * ``False`` → identity
            * ``nn.Module`` → deep-copied custom module
    """

    default_act: nn.Module = nn.SiLU()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        activation: Union[bool, nn.Module] = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        # First 1 x 1 projection producing 2·hidden_channels, then split
        act1 = _clone_act(self.default_act, activation)
        self.conv1 = ConvBNActivation(
            in_channels,
            2 * hidden_channels,
            kernel_size=1,
            stride=1,
            activation=act1,
        )

        # Bottleneck stack operates on the first half (hidden_channels)
        self.bottlenecks = nn.Sequential(
            *(
                BottleneckBlock(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    groups=groups,
                    kernel_sizes=((3, 3), (3, 3)),
                    expansion=1.0,
                )
                for _ in range(num_blocks)
            )
        )

        # Final 1 x 1 to fuse concatenation
        act2 = _clone_act(self.default_act, activation)
        self.conv2 = ConvBNActivation(
            2 * hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            activation=act2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x → Conv1x1 → split → bottleneck stack → concat → Conv1x1."""
        y1, y2 = self.conv1(x).chunk(2, dim=1)
        y1 = self.bottlenecks(y1)
        return self.conv2(torch.cat((y1, y2), dim=1))


class CSPDualConvFastBottleneckBlock(nn.Module):
    """Fast CSP bottleneck with expandable split-stack-concat pattern.

    The design follows the *C2f* pattern used in recent YOLO variants,
    but re-implemented with Google-style, ONNX-friendly code:

    1. **Projection** - 1 x 1 conv producing *2·h* channels
       ``y0, y1 = split(conv1x1(x))``
    2. **Stack** - repeatedly apply *n* Bottleneck blocks to *y1*,
       appending each output to the list.
    3. **Fusion** - concatenate all tensors and fuse with a final 1 x 1 conv.

    Args:
        in_channels (int):   Input feature channels.
        out_channels (int):  Output feature channels.
        num_blocks (int, optional): Number of inner :class:`Bottleneck`
            layers. Defaults to ``1``.
        shortcut (bool, optional): Enable shortcut in each Bottleneck.
            Defaults to ``False`` (matches original *C2f*).
        groups (int, optional): Convolution groups for Bottlenecks.
            Defaults to ``1``.
        expansion (float, optional): Hidden-channel expansion ratio.
            Defaults to ``0.5``.
        activation (bool | nn.Module, optional): Activation selector.

            * ``True`` → fresh SiLU (default)
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
        num_blocks: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion: float = 0.5,
        activation: Union[bool, nn.Module] = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        act1 = _clone_act(self.default_act, activation)
        self.conv1 = ConvBNActivation(
            in_channels,
            2 * hidden_channels,
            kernel_size=1,
            stride=1,
            activation=act1,
        )

        # Dynamically sized output channels: 2 + num_blocks splits
        act2 = _clone_act(self.default_act, activation)
        self.conv2 = ConvBNActivation(
            (2 + num_blocks) * hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            activation=act2,
        )

        # Inner Bottleneck stack (ModuleList for iterative access)
        self.blocks = nn.ModuleList(
            BottleneckBlock(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                groups=groups,
                kernel_sizes=((3, 3), (3, 3)),
                expansion=1.0,
            )
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using :py:meth:`torch.Tensor.chunk`."""
        parts = list(self.conv1(x).chunk(2, dim=1))  # [y0, y1]

        for blk in self.blocks:
            parts.append(blk(parts[-1]))

        return self.conv2(torch.cat(parts, dim=1))


class CSPTripleConvBottleneckBlock(nn.Module):
    """CSP bottleneck with **three** point-wise convolutions.

    The flow is::

        y1 = Conv1x1(x)              # projection branch
        y1 = Bottleneck x n (y1)     # transform branch
        y2 = Conv1x1(x)              # skip branch
        out = Conv1x1(cat(y1, y2))   # fusion

    Args:
        in_channels (int):   Input feature channels.
        out_channels (int):  Output feature channels.
        num_blocks (int, optional): Number of internal :class:`Bottleneck`
            layers. Defaults to ``1``.
        shortcut (bool, optional): Enable shortcut inside each Bottleneck.
            Defaults to ``True``.
        groups (int, optional): Convolution groups for Bottlenecks.
            Defaults to ``1``.
        expansion (float, optional): Hidden-channel expansion ratio.
            Defaults to ``0.5``.
        activation (bool | nn.Module, optional): Activation selector.

            * ``True`` → new SiLU instance (default)
            * ``False`` → identity (no activation)
            * ``nn.Module`` → user-supplied module (deep-copied)

    Example:
        >>> blk = CSPTripleConvBottleneckBlock(128, 256, num_blocks=3)
        >>> y = blk(torch.randn(1, 128, 28, 28))
    """

    default_act: nn.Module = nn.SiLU()

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        activation: Union[bool, nn.Module] = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        # Point-wise convolutions before and after fusion.
        self.conv_proj = ConvBNActivation(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            activation=_clone_act(self.default_act, activation),
        )
        self.conv_skip = ConvBNActivation(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            activation=_clone_act(self.default_act, activation),
        )
        self.conv_fuse = ConvBNActivation(
            2 * hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            activation=_clone_act(self.default_act, activation),
        )

        # Sequential Bottleneck stack operating on *hidden_channels*.
        self.bottlenecks = nn.Sequential(
            *(
                BottleneckBlock(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    groups=groups,
                    kernel_sizes=((3, 3), (3, 3)),
                    expansion=1.0,
                )
                for _ in range(num_blocks)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x → projection → Bottleneck stack + skip → concat → fusion."""
        y1 = self.bottlenecks(self.conv_proj(x))
        y2 = self.conv_skip(x)
        return self.conv_fuse(torch.cat((y1, y2), dim=1))


class CSPTripleConvKernelBlock(CSPTripleConvBottleneckBlock):
    """C3-k: C3 architecture with a *k x k* inner kernel.

    This class re-uses the overall flow of :class:`CSPTripleConvBottleneckBlock`
    (projection → Bottleneck stack → fusion) but allows the second convolution
    inside each Bottleneck to use an arbitrary square kernel *k x k*.

    Only the Bottleneck stack is replaced; all projection/skip/fusion layers
    remain identical to the parent implementation.

    Args:
        in_channels (int):   Input feature channels.
        out_channels (int):  Output feature channels.
        num_blocks (int, optional): Number of inner Bottleneck blocks.
            Defaults to ``1``.
        kernel_size (int, optional): Square kernel size *k* for the inner
            Bottlenecks. Defaults to ``3``.
        shortcut (bool, optional): Enable shortcut in each Bottleneck.
            Defaults to ``True``.
        groups (int, optional): Convolution groups for Bottlenecks.
            Defaults to ``1``.
        expansion (float, optional): Hidden-channel expansion ratio.
            Defaults to ``0.5``.
        activation (bool | nn.Module, optional): Activation selector.
            * ``True`` → fresh SiLU (default)
            * ``False`` → identity
            * ``nn.Module`` → user-supplied (deep-copied)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int = 1,
        kernel_size: int = 3,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        activation: Union[bool, nn.Module] = True,
    ) -> None:
        # Build base C3 structure (projection / skip / fusion).
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            shortcut=shortcut,
            groups=groups,
            expansion=expansion,
            activation=activation,
        )

        # -----------------------------------------------------------------
        # Replace the parent’s Bottleneck stack with a custom-kernel stack.
        # -----------------------------------------------------------------
        hidden_channels = int(out_channels * expansion)

        self.bottlenecks = nn.Sequential(
            *(
                BottleneckBlock(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    groups=groups,
                    kernel_sizes=(kernel_size, kernel_size),
                    expansion=1.0,
                )
                for _ in range(num_blocks)
            )
        )


class CSPKernelMixFastBottleneckBlock(CSPDualConvFastBottleneckBlock):
    """Fast CSP bottleneck (*C3k2*) with optional kernel-mix inner blocks.

    This block inherits the projection/split/fusion logic from
    :class:`CSPDualConvFastBottleneckBlock` (formerly *C2f*) and lets you pick
    the inner stack implementation:

    * **CSPTripleConvKernelBlock** (*C3k*) — when ``use_c3k`` is ``True``
    * **BottleneckBlock**                  — otherwise

    Args:
        in_channels (int):   Input feature channels.
        out_channels (int):  Output feature channels.
        num_blocks (int, optional): Number of inner blocks. Defaults ``1``.
        use_c3k (bool, optional): Use *C3k* inner blocks if ``True``.
            Defaults ``False``.
        expansion (float, optional): Hidden-channel expansion ratio.
            Defaults ``0.5``.
        groups (int, optional): Convolution groups for inner blocks.
            Defaults ``1``.
        shortcut (bool, optional): Enable shortcut inside each inner block.
            Defaults ``True``.
        activation (bool | nn.Module, optional): Activation selector.

            * ``True`` → fresh SiLU (default)
            * ``False`` → identity
            * ``nn.Module`` → user-supplied (deep-copied)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int = 1,
        use_c3k: bool = False,
        expansion: float = 0.5,
        groups: int = 1,
        shortcut: bool = True,
        activation: Union[bool, nn.Module] = True,
    ) -> None:
        # Initialize parent → sets up projection / fusion layers.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            shortcut=shortcut,
            groups=groups,
            expansion=expansion,
            activation=activation,
        )

        hidden_channels = int(out_channels * expansion)

        # Replace parent-defined inner stack with the desired block type.
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_c3k:
                block = CSPTripleConvKernelBlock(
                    hidden_channels,
                    hidden_channels,
                    num_blocks=2,
                    shortcut=shortcut,
                    groups=groups
                )
            else:
                block = BottleneckBlock(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    groups=groups
                )
            self.blocks.append(block)


class CSPDualPSAStackBlock(nn.Module):
    """C2-PSA: dual-branch CSP block that stacks PSA sub-modules.

    Workflow::

        y0, y1 = split(Conv1x1(x))        # 1. projection & split
        y1 = PSA x n (y1)                 # 2. stacked PSA blocks
        out = Conv1x1(cat(y0, y1))        # 3. fusion

    Args:
        in_channels (int):   Input/Output feature channels *C* (must match).
        num_blocks (int):    Number of :class:`PositionSensitiveAttention`
            stacked on the second branch.
        expansion (float):   Hidden-channel expansion ratio (0 < e ≤ 1).
        attn_ratio (float):  Key/query dimensionality ratio in PSA.
        activation (bool | nn.Module, optional): Activation selector for all
            1 x 1 convolutions inside the block.

            * ``True`` → new SiLU (default)
            * ``False`` → identity
            * ``nn.Module`` → user-supplied (deep-copied)
    """

    default_act: nn.Module = nn.SiLU()

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_channels: int,
        *,
        num_blocks: int = 1,
        expansion: float = 0.5,
        attn_ratio: float = 0.5,
        activation: Union[bool, nn.Module] = True,
    ) -> None:
        super().__init__()

        self.hidden = hidden = int(in_channels * expansion)

        # --- 1 x 1 projection & split ----------------------------------- #
        self.conv_proj = ConvBNActivation(
            in_channels,
            2 * hidden,
            kernel_size=1,
            stride=1,
            activation=_clone_act(self.default_act, activation),
        )

        # --- PSA stack on the second split ------------------------------ #
        self.psa_stack = nn.Sequential(
            *(
                PositionSensitiveAttention(
                    channels=hidden,
                    attn_ratio=attn_ratio,
                    num_heads=max(1, hidden // 64)
                )
                for _ in range(num_blocks)
            )
        )

        # --- 1 x 1 fusion ------------------------------------------------ #
        self.conv_fuse = ConvBNActivation(
            2 * hidden,
            in_channels,
            kernel_size=1,
            stride=1,
            activation=_clone_act(self.default_act, activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x → split → PSA stack → concat → fuse."""
        y0, y1 = self.conv_proj(x).split((self.hidden, self.hidden), dim=1)
        y1 = self.psa_stack(y1)
        return self.conv_fuse(torch.cat((y0, y1), dim=1))
