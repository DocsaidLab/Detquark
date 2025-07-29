from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..blocks import DistributionFocalLoss
from ..utils import bbox2dist, bbox_iou


class BBoxLoss(nn.Module):
    """Compute IoU-based loss and Distribution-Focal Loss (DFL) for boxes.

    This criterion pairs **localization quality** (IoU/CIoU) with
    **distribution regression quality** (DFL) so the model learns to place
    boxes accurately *and* predict precise per-side distances.

    Args:
        reg_max: Number of discrete distance bins *R* used by the DFL head.
            When ``reg_max == 1`` the model regresses raw distances and the
            DFL term is skipped.

    Forward inputs
    --------------
    * ``pred_dist``        - ``(N, 4 x R)`` logits for DFL distances.
    * ``pred_bboxes``      - ``(N, 4)`` decoded boxes from model.
    * ``anchor_points``    - ``(N, 2)`` grid centres.
    * ``target_bboxes``    - ``(N, 4)`` ground-truth boxes (x1, y1, x2, y2).
    * ``target_scores``    - ``(N, C)`` or ``(N, 1)`` classification scores
                              used as soft weights.
    * ``target_scores_sum``- Scalar normaliser ``target_scores.sum()``.
    * ``fg_mask``          - ``(N,)`` boolean mask for foreground anchors.

    Returns:
        Tuple ``(loss_iou, loss_dfl)``; if *reg_max == 1* the second term is
        zero.

    Note:
        This implementation expects that *bbox_iou* already supports the
        `ciou=True` keyword and that *bbox2dist* clips distances to
        ``[0, reg_max - 1)``.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max: int = reg_max
        self.dfl_loss: Optional[DistributionFocalLoss] = (
            DistributionFocalLoss(reg_max) if reg_max > 1 else None
        )

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute localisation (CIoU) loss and optional DFL loss."""
        # ── IoU / CIoU loss ────────────────────────────────────────────────
        weight = target_scores.sum(dim=-1, keepdim=True)[fg_mask]  # (M, 1)
        iou = bbox_iou(  # CIoU variant
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            xywh=False,
            ciou=True,
        )
        loss_iou = ((1.0 - iou) * weight).sum() / (target_scores_sum + 1e-9)

        # ── DFL loss (optional) ────────────────────────────────────────────
        loss_dfl = torch.zeros(1, device=pred_dist.device)
        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.reg_max - 1)
            loss_raw = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.reg_max),  # logits  (..., R)
                target_ltrb[fg_mask].view(-1),              # targets (...,)
                reduction="none",
            ).view_as(weight)  # reshape back to (M, 1)
            loss_dfl = (loss_raw * weight).sum() / (target_scores_sum + 1e-9)

        return loss_iou, loss_dfl
