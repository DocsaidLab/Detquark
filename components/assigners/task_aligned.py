from __future__ import annotations

"""Task-Aligned Assigner
========================
A PyTorch implementation of the **Task-Aligned Assigner** (TAL) used in modern
anchor-based/anchor-free object detectors.

This module pairs each anchor point with at most one ground-truth (GT) object
according to the _task-aligned metric_ (classification - IoU).  It produces
labels, boxes, and soft scores required by loss functions such as **BBoxLoss**.
"""

from typing import Tuple

import torch
import torch.nn as nn

from ..utils import bbox_iou

__all__ = ["TaskAlignedAssigner"]


class TaskAlignedAssigner(nn.Module):
    """Assign ground-truth boxes to feature-map anchors via Task-Aligned metric.

    Args:
        topk: Number of top-k candidate anchors per GT used during the coarse
            candidate search.
        num_classes: Total number of object classes in the dataset (without
            background).
        alpha: Exponent applied to the classification score in the metric.
        beta: Exponent applied to IoU in the metric.
        eps: Small constant for numerical stability when normalising metrics.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        *,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1.0e-9,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # The following attributes are set lazily inside :pyfunc:`forward` for
        # convenience; they are *not* buffers and hence excluded from state-dict.
        self._bs: int | None = None
        self._n_max_boxes: int | None = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign anchors to GT boxes and build regression/cls targets.

        Args:
            pd_scores: Raw classification scores ``(B, N, C)`` from the model.
            pd_bboxes: Predicted boxes (decoded) ``(B, N, 4)``.
            anc_points: Anchor/grid centres ``(N, 2)`` in image space.
            gt_labels: Integer GT labels ``(B, M, 1)``.
            gt_bboxes: GT boxes ``(B, M, 4)`` in ``(x1, y1, x2, y2)``.
            mask_gt: Boolean tensor marking valid GT entries ``(B, M, 1)``.

        Returns:
            Tuple consisting of
            ``(target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx)``.
        """
        self._bs = int(pd_scores.shape[0])
        self._n_max_boxes = int(gt_bboxes.shape[1])
        device = gt_bboxes.device

        if self._n_max_boxes == 0:
            # No GT in batch — return dummy tensors.
            return (
                # labels = background
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]).bool(),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.long),
            )

        # Main logic is delegated to the private helper to optionally enable
        # CPU fallback when out-of-memory.
        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            cpu_tensors = [t.cpu() for t in (
                pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    # ------------------------------------------------------------------
    # Internal helpers (prefixed with underscore)
    # ------------------------------------------------------------------
    def _forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inner implementation — identical signature to :pyfunc:`forward`."""
        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self._select_highest_overlaps(
            mask_pos, overlaps, self._n_max_boxes
        )

        # ------------------------------------------------------------------
        # Build training targets
        # ------------------------------------------------------------------
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Metric normalisation (scale each anchor's score by GT-level stats)
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(
            dim=-1, keepdim=True)  # (B, M, 1)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-
                                                  1, keepdim=True)  # (B, M, 1)
        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(dim=-2, keepdim=True)  # (B, 1, N)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    # ..................................................................
    # Candidate selection helpers
    # ..................................................................
    def _get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return positive mask, alignment metric, and IoU overlaps."""
        mask_in_gts = self._select_candidates_in_gts(
            anc_points, gt_bboxes)  # (B, M, N)

        # Alignment metric (classification **alpha × IoU ** beta)
        align_metric, overlaps = self._get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts & mask_gt.bool()
        )

        # Top-k filtering per GT
        mask_topk = self._select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )

        mask_pos = mask_topk & mask_in_gts & mask_gt.bool()
        return mask_pos, align_metric, overlaps

    def _get_box_metrics(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute alignment metric and IoU overlaps for each GT-anchor pair."""
        b, m, n = mask_gt.shape
        device, dtype_s, dtype_b = pd_scores.device, pd_scores.dtype, pd_bboxes.dtype

        overlaps = torch.zeros((b, m, n), dtype=dtype_b, device=device)
        bbox_scores = torch.zeros((b, m, n), dtype=dtype_s, device=device)

        # Build indices for advanced indexing: (2, B, M)
        batch_ind = torch.arange(b, device=device).view(
            b, 1).expand(-1, m)  # (B, M)
        cls_ind = gt_labels.squeeze(-1)  # (B, M)

        # Gather classification scores for each GT class across all anchors
        bbox_scores[mask_gt] = pd_scores[batch_ind, :, cls_ind][mask_gt]

        # Gather IoU between each GT and all anchors (mask to save memory)
        pd_boxes_expanded = pd_bboxes.unsqueeze(
            1).expand(-1, m, -1, -1)[mask_gt]
        gt_boxes_expanded = gt_bboxes.unsqueeze(
            2).expand(-1, -1, n, -1)[mask_gt]
        overlaps[mask_gt] = self._iou_calculation(
            gt_boxes_expanded, pd_boxes_expanded)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    @staticmethod
    def _iou_calculation(gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        """Compute CIoU between two equally-shaped bbox tensors."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, ciou=True).squeeze(-1).clamp_(0)

    # .................................................................
    # Top-k selection (vectorised, no Python loop)
    # .................................................................
    def _select_topk_candidates(self, metrics: torch.Tensor, *, topk_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return boolean mask of top-k anchors for each GT based on *metrics*."""
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            # If caller didn't supply valid-GT mask, treat any metric > eps as candidate.
            topk_mask = (topk_metrics.max(
                dim=-1, keepdim=True).values > self.eps).expand_as(topk_idxs)

        selected = torch.zeros_like(metrics, dtype=torch.bool)
        selected.scatter_(-1, topk_idxs, topk_mask)
        return selected

    # .................................................................
    # Target construction helpers
    # .................................................................
    def _get_targets(
        self,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n = fg_mask.shape
        device = gt_labels.device

        # Build global GT index = batch_idx * M + gt_idx
        offset = (torch.arange(b, device=device)
                  * self._n_max_boxes).view(b, 1)
        global_gt_idx = target_gt_idx + offset  # (B, N)

        # Gather labels & boxes
        flat_labels = gt_labels.long().flatten()  # (B*M, 1)
        # (B, N, 1) -> squeeze later
        target_labels = flat_labels[global_gt_idx]

        flat_boxes = gt_bboxes.view(-1, 4)
        target_bboxes = flat_boxes[global_gt_idx]  # (B, N, 4)

        # One-hot scores (faster than F.one_hot) then mask by fg
        target_scores = torch.zeros(
            (b, n, self.num_classes), dtype=torch.float32, device=device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1.0)
        target_scores *= fg_mask.unsqueeze(-1)

        return target_labels.squeeze(-1), target_bboxes, target_scores

    # .................................................................
    # Static-utility helpers
    # .................................................................
    @staticmethod
    def _select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, *, eps: float = 1.0e-9) -> torch.Tensor:
        """Return mask of anchors whose centres lie inside any GT bbox."""
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape

        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, dim=2)  # (B*M, 1, 2)
        deltas = torch.cat(
            (xy_centers[None] - lt, rb - xy_centers[None]), dim=2)
        deltas = deltas.view(bs, n_boxes, n_anchors, 4)
        return deltas.amin(dim=3).gt_(eps)  # (B, M, N)

    @staticmethod
    def _select_highest_overlaps(
        mask_pos: torch.Tensor,
        overlaps: torch.Tensor,
        n_max_boxes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve conflicts where multiple GTs match the same anchor."""
        fg_mask = mask_pos.sum(dim=-2)  # (B, N)
        if fg_mask.max() > 1:
            multi_gt_mask = (fg_mask.unsqueeze(1) > 1)  # (B, 1, N)
            max_overlaps_idx = overlaps.argmax(
                dim=1, keepdim=True)  # (B, 1, N)

            is_max = torch.zeros_like(mask_pos, dtype=mask_pos.dtype)
            is_max.scatter_(1, max_overlaps_idx, 1)
            mask_pos = torch.where(multi_gt_mask, is_max, mask_pos).float()
            fg_mask = mask_pos.sum(dim=-2)

        target_gt_idx = mask_pos.argmax(dim=-2)  # (B, N)
        return target_gt_idx, fg_mask, mask_pos
