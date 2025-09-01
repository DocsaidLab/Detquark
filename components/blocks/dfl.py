from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DFLIntegral",
    "DistributionFocalLoss",
]


class DFLIntegral(nn.Module):
    """Integral operator for Distribution-Focal Loss (DFL).

    This module computes the integral of a discrete probability distribution
    over a range of regression bins using a fixed convolutional kernel.

    Attributes:
        reg_max (int): Number of discrete bins in the distribution.
        conv (nn.Conv2d): Fixed 1x1 convolution layer implementing the integral
            operator weights.
    """

    def __init__(self, reg_max: int = 16) -> None:
        """Initializes the DFLIntegral operator.

        Args:
            reg_max (int): Number of discrete bins in the distribution.
                Defaults to 16.
        """
        super().__init__()
        self.reg_max: int = reg_max
        # Fixed 1x1 convolution with weights equal to [0, 1, ..., reg_max-1]
        self.conv = nn.Conv2d(reg_max, 1, kernel_size=1, bias=False)
        self.conv.requires_grad_(False)

        # Initialize convolution weights to [0,1,...,reg_max-1] shape (1, reg_max, 1, 1)
        weight = \
            torch.arange(reg_max, dtype=torch.float).view(1, reg_max, 1, 1)
        self.conv.weight.data.copy_(weight)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes the integral of the probability distribution.

        Args:
            logits (torch.Tensor):
                Input tensor of shape (batch_size, 4 * reg_max, num_anchors)
                representing raw distribution logits.

        Returns:
            torch.Tensor:
                Tensor of shape (batch_size, 4, num_anchors) containing
                the predicted continuous regression values.
        """
        batch_size, _, num_anchors = logits.shape
        # Reshape to (batch_size, 4, reg_max, num_anchors) then softmax over bins
        prob = (
            logits
            .view(batch_size, 4, self.reg_max, num_anchors)
            .transpose(1, 2)
            .softmax(dim=1)
        )
        # Apply fixed convolution to compute expectation over bins
        integral = self.conv(prob).view(batch_size, 4, num_anchors)
        return integral


class DistributionFocalLoss(nn.Module):
    """Distribution- Focal Loss (DFL) criterion.

    Implements the *generalised focal loss* described in
    *Li et al., “Generalized Focal Loss” (CVPR 2021)* and later refined in
    *“A Center- Assisted Flake Detection Network” (IEEE TITS 2022, §III- B)*.

    Given continuous distance targets *t* ∈ [0, R) and discrete logits
    *p* ∈ ℝᴿ, the loss softly assigns *t* to its neighbouring integer bins
    ``⌊t⌋`` and ``⌈t⌉`` with linear weights, then sums two cross- entropy terms.

    Args:
        reg_max: Number of discrete bins *R* per side (e.g. 16).

    Shape conventions
    -----------------
    * ``pred``   - ``(..., R)`` logits for one side (left / top / …).
    * ``target`` - ``(...,)`` continuous ground- truth distances.
      The leading dimensions *must* match those of ``pred``.

    Returns:
        Loss value reduced as specified by *reduction*.

    Example
    -------
    >>> loss_fn = DistributionFocalLoss(reg_max=16)
    >>> pred = torch.randn(32, 16)      # logits
    >>> target = torch.rand(32) * 16    # continuous distances
    >>> loss = loss_fn(pred, target)    # scalar
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max: int = reg_max

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> torch.Tensor:
        """
        Args:
            pred: Discrete logits of shape ``(..., reg_max)``.
            target: Continuous distances of identical leading shape ``(...,)``.
            reduction: ``"none"``, ``"mean"``, or ``"sum"``.  Default ``"mean"``.

        Returns:
            Tensor containing the unreduced loss (``"none"``) or a scalar.

        Raises:
            ValueError: If the last dimension of *pred* is not ``reg_max``.
        """
        if pred.shape[-1] != self.reg_max:
            raise ValueError(
                f"pred last dimension must be {self.reg_max}, got {pred.shape[-1]}."
            )

        # Clamp target to valid range [0, R- 1).
        target = target.clamp_(0.0, self.reg_max - 1 - 0.01)

        # Left / right integer bins and linear weights.
        tl = target.long()            # target left bin
        tr = (tl + 1)                 # target right bin
        wl = tr - target              # weight for left bin
        wr = 1 - wl                   # weight for right bin

        # Flatten everything except logits dimension for CE.
        ce_left = F.cross_entropy(
            pred, tl.view(-1), reduction="none").view_as(tl)
        ce_right = F.cross_entropy(
            pred, tr.view(-1), reduction="none").view_as(tl)

        loss = ce_left * wl + ce_right * wr

        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()

        return loss
