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

    Converts discrete per-side logits into continuous distances via the
    expectation ∑ p x bin_idx, where *bin_idx* ∈ {0,‥, R-1}.

    Typical workflow
    ----------------
    >>> logits = torch.randn(B, 4 * 16, A)         # model output
    >>> dist    = DFLIntegral(16)(logits)          # (B, 4, A) continuous
    >>> boxes   = dist2bbox(dist, anchors)         # decode to boxes
    """

    def __init__(self, reg_max: int = 16) -> None:
        """
        Args:
            reg_max: Number of discrete bins *R* for each side
                     (same value used during training loss computation).
        """
        super().__init__()
        self.reg_max: int = reg_max

        # Register a non-trainable buffer [0, 1, ..., R-1] for dot-product.
        proj = torch.linspace(0, reg_max - 1, reg_max).view(1, 1, reg_max, 1)
        self.register_buffer("project", proj, persistent=False)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape ``(B, 4 x R, A)``, where *R = reg_max*.

        Returns:
            Tensor of shape ``(B, 4, A)`` — expected distances per side.
        """
        b, _, a = logits.shape
        if logits.shape[1] % self.reg_max != 0:
            raise ValueError(
                f"Channel dimension ({logits.shape[1]}) must be divisible by "
                f"reg_max ({self.reg_max})."
            )

        # Reshape → (B, 4, R, A) then softmax over R
        prob = logits.view(b, 4, self.reg_max, a).softmax(dim=2)

        # Expectation: ∑ p_k · k  (broadcast dot product with self.project)
        return (prob * self.project).sum(dim=2)  # (B, 4, A)


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
        target = target.clamp_(0.0, self.reg_max - 1 - 1e-3)

        # Left / right integer bins and linear weights.
        tl = target.floor().long()          # target left bin
        tr = (tl + 1).clamp(max=self.reg_max - 1)  # target right bin
        wl = (tr.float() - target)          # weight for left bin
        wr = 1.0 - wl                       # weight for right bin

        # Flatten everything except logits dimension for CE.
        ce_left = F.cross_entropy(
            pred.view(-1, self.reg_max), tl.view(-1), reduction="none"
        ).view_as(tl)
        ce_right = F.cross_entropy(
            pred.view(-1, self.reg_max), tr.view(-1), reduction="none"
        ).view_as(tl)

        loss = ce_left * wl + ce_right * wr  # (...,)

        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()

        return loss  # reduction == "none"
