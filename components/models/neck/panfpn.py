from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from ...blocks import CSPKernelMixFastBottleneckBlock
from ...layers import ConvBNActivation

__all__ = [
    "PANFPNBlock",
    "PANFPN",
]


# --------------------------------------------------------------------------- #
# PANFPNBlock                                                                 #
# --------------------------------------------------------------------------- #
class PANFPNBlock(nn.Module):
    """Single-stage PAN-FPN fusion block.

    Implements a single pass of top-down and bottom-up feature fusion:
    First performs top-down fusion (Upsample → Concat → C3k2),
    then bottom-up fusion (Conv stride=2 → Concat → C3k2).

    Args:
        in_channels_list (Sequence[int]): List of channel counts from low to high
            level features (e.g., P3 → Pn), e.g., [256, 512, 1024].
        repeat (int, optional): Number of repeated bottleneck blocks inside each
            `CSPKernelMixFastBottleneckBlock`. Default is 2.
        enable_c3k_last (bool, optional): Enable `use_c3k=True` only on the final
            bottom-up C3k2 block (usually on the highest level, e.g., P5/32),
            following Ultralytics design. Default is True.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        repeat: int = 2,
        enable_c3k_last: bool = True,
    ) -> None:
        super().__init__()

        self.levels: int = len(in_channels_list)
        if self.levels < 2:
            raise ValueError(
                "PAN-FPN requires at least 2 feature levels (e.g., P3 and P4).")

        # ───── Top-down: Upsample → Concat → C3k2 ─────
        self.upsamples = nn.ModuleList([
            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            ) for _ in range(self.levels - 1)
        ])

        self.td_blocks = nn.ModuleList()
        for hi in reversed(range(1, self.levels)):  # from higher to lower (Pn → P3)
            lo = hi - 1
            self.td_blocks.append(
                CSPKernelMixFastBottleneckBlock(
                    in_channels_list[hi] + in_channels_list[lo],
                    in_channels_list[lo],
                    num_blocks=repeat,
                    use_c3k=False,  # Top-down blocks always disable use_c3k
                )
            )

        # ───── Bottom-up: Conv stride=2 → Concat → C3k2 ─────
        self.downsamples = nn.ModuleList()
        self.bu_blocks = nn.ModuleList()
        for lo in range(self.levels - 1):  # from lower to higher (P3 → Pn-1)
            hi = lo + 1
            self.downsamples.append(
                ConvBNActivation(
                    in_channels_list[lo],
                    in_channels_list[lo],
                    kernel_size=3,
                    stride=2,
                )
            )
            self.bu_blocks.append(
                CSPKernelMixFastBottleneckBlock(
                    in_channels=in_channels_list[lo] + in_channels_list[hi],
                    out_channels=in_channels_list[hi],
                    num_blocks=repeat,
                    use_c3k=enable_c3k_last and (hi == self.levels - 1),
                )
            )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform one stage of bidirectional feature fusion.

        Args:
            feats (List[torch.Tensor]): Input feature maps ordered according to
                `in_channels_list` from low to high level (P3 → Pn).

        Returns:
            List[torch.Tensor]: Fused feature maps in the same order.
        """
        if len(feats) != self.levels:
            raise ValueError(
                f"Expected {self.levels} feature maps, but got {len(feats)}."
            )

        feats = list(feats)  # Avoid in-place modification of upstream tensors

        # Top-down fusion: from highest to lowest level (Pn → P3)
        for idx, td_block in enumerate(self.td_blocks):
            hi = self.levels - 1 - idx  # e.g., 2, 1, ...
            lo = hi - 1                 # e.g., 1, 0, ...
            fused = torch.cat(
                [self.upsamples[hi - 1](feats[hi]), feats[lo]], dim=1
            )
            feats[lo] = td_block(fused)

        # Bottom-up fusion: from lowest to highest level (P3 → Pn)
        for lo, (down, bu_block) in enumerate(zip(self.downsamples, self.bu_blocks)):
            hi = lo + 1
            fused = torch.cat([down(feats[lo]), feats[hi]], dim=1)
            feats[hi] = bu_block(fused)

        return feats


class PANFPN(nn.Module):
    """Stacked PAN-FPN neck with customizable input and output feature levels.

    Args:
        in_channels_list (List[int]): Channel counts of backbone feature maps.
        in_indices (List[int]): Indices of backbone feature levels to use (sorted
            from low resolution to high resolution).
        n_blocks (int, optional): Number of stacked `PANFPNBlock` layers. Default is 1.
        repeat (int, optional): Passed to `PANFPNBlock.repeat` to control depth of
            C3k2 blocks. Default is 2.
        enable_c3k_last (bool, optional): Whether to enable `use_c3k=True` on the
            final bottom-up C3k2 block. Default is True.
        out_indices (List[int] | None, optional): Indices of output feature levels
            relative to `in_indices`. Defaults to all outputs.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        in_indices: List[int],
        n_blocks: int = 1,
        repeat: int = 2,
        enable_c3k_last: bool = True,
        out_indices: List[int] | None = None,
    ) -> None:
        super().__init__()

        if n_blocks < 1:
            raise ValueError("`n_blocks` must be >= 1.")
        if sorted(in_indices) != in_indices:
            raise ValueError(
                "`in_indices` must be sorted from low to high resolution.")

        self.in_indices = in_indices
        self.sel_channels = [in_channels_list[i] for i in in_indices]
        self.out_indices = out_indices or list(range(len(in_indices)))

        if max(self.out_indices) >= len(in_indices):
            raise ValueError("`out_indices` exceeds range of `in_indices`.")

        # Create stacked PANFPNBlocks
        self.blocks = nn.ModuleList(
            [
                PANFPNBlock(
                    in_channels_list=self.sel_channels,
                    repeat=repeat,
                    enable_c3k_last=enable_c3k_last,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            feats (List[torch.Tensor]): Backbone output feature maps in order
                corresponding to `in_channels_list`.

        Returns:
            List[torch.Tensor]: PAN-FPN features at specified output indices.
        """
        x = [feats[i] for i in self.in_indices]  # Select input levels
        for block in self.blocks:
            x = block(x)
        return [x[i] for i in self.out_indices]

    @property
    def out_channels(self) -> List[int]:
        """Channels of output feature maps (aligned with `out_indices`)."""
        return [self.sel_channels[i] for i in self.out_indices]
