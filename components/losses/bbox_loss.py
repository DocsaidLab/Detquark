from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..blocks import DistributionFocalLoss
from ..utils import bbox2dist, bbox_iou


class BBoxLoss(nn.Module):

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DistributionFocalLoss(reg_max) if reg_max > 1 else None

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
        weight = target_scores.sum(dim=-1, keepdim=True)[fg_mask]
        iou = bbox_iou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            xywh=False,
            ciou=True,
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        loss_dfl = torch.zeros((), device=pred_dist.device)
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.reg_max - 1
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.reg_max),
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
