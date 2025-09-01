from __future__ import annotations

from warnings import warn

import torch
import torch.nn as nn

from ..utils import bbox_iou


class TaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner (TAL) for anchor-based object detection.

    This module selects positive anchor candidates for training
    by jointly considering classification confidence and localization IoU.

    Attributes:
        topk (int): Number of top-k candidates to consider per ground truth.
        num_classes (int): Total number of target object classes.
        alpha (float): Power factor for classification score weight.
        beta (float): Power factor for IoU score weight.
        eps (float): Small constant for numerical stability.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9
    ) -> None:
        """Initialize the TaskAlignedAssigner module.

        Args:
            topk (int, optional): Number of top candidates to keep for each GT.
            num_classes (int, optional): Number of classes in the dataset.
            alpha (float, optional): Weight exponent for classification scores.
            beta (float, optional): Weight exponent for IoU overlaps.
            eps (float, optional): Epsilon to avoid division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Main forward entry point for TaskAlignedAssigner.

        This function computes assignment results for a batch of predictions,
        performing fallback to CPU if CUDA runs out of memory.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores, shape (B, H*W, C).
            pd_bboxes (torch.Tensor): Predicted bounding boxes, shape (B, H*W, 4).
            anc_points (torch.Tensor): Anchor centers, shape (H*W, 2).
            gt_labels (torch.Tensor): Ground truth class labels, shape (B, N, 1).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (B, N, 4).
            mask_gt (torch.Tensor): Boolean mask indicating valid GTs, shape (B, N).

        Returns:
            target_labels (torch.Tensor): Assigned class labels, shape (B, H*W).
            target_bboxes (torch.Tensor): Assigned GT boxes, shape (B, H*W, 4).
            target_scores (torch.Tensor): One-hot encoded target scores, shape (B, H*W, C).
            fg_mask (torch.Tensor): Foreground mask, shape (B, H*W).
            align_metric (torch.Tensor): Alignment metric score, shape (B, H*W).
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            # No GTs: return background class and zero targets
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),  # labels
                torch.zeros_like(pd_bboxes),  # boxes
                torch.zeros_like(pd_scores),  # scores
                torch.zeros_like(pd_scores[..., 0]),  # fg_mask
                torch.zeros_like(pd_scores[..., 0]),  # align_metric
            )

        try:

            return self._forward(
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes,
                mask_gt
            )

        except torch.cuda.OutOfMemoryError:

            warn("⚠️ Falling back to CPU due to CUDA OOM in TaskAlignedAssigner.")

            cpu_inputs = [
                x.cpu()
                for x in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
            ]
            result = self._forward(*cpu_inputs)

            return tuple(x.to(device) for x in result)

    def _forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Internal logic for forward pass of TaskAlignedAssigner.

        Args:
            pd_scores (torch.Tensor):
                Predicted class scores, shape (B, H*W, C).
            pd_bboxes (torch.Tensor):
                Predicted bounding boxes, shape (B, H*W, 4).
            anc_points (torch.Tensor):
                Anchor centers, shape (H*W, 2).
            gt_labels (torch.Tensor):
                Ground truth class labels, shape (B, N, 1).
            gt_bboxes (torch.Tensor):
                Ground truth bounding boxes, shape (B, N, 4).
            mask_gt (torch.Tensor):
                Boolean mask indicating valid GTs, shape (B, N).

        Returns:
            target_labels (torch.Tensor):
                Assigned class labels, shape (B, H*W).
            target_bboxes (torch.Tensor):
                Assigned GT boxes, shape (B, H*W, 4).
            target_scores (torch.Tensor):
                One-hot encoded scores, shape (B, H*W, C).
            fg_mask (torch.Tensor):
                Foreground mask, shape (B, H*W), dtype=torch.bool.
            target_gt_idx (torch.Tensor):
                Assigned GT indices, shape (B, H*W).
        """

        # Get candidate mask and task-aligned metric
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        # Resolve multiple GT assignments for same anchor
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )

        # Generate target labels, boxes, scores
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Normalize alignment metric and refine scores
        align_metric *= mask_pos

        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)

        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(dim=-2).unsqueeze(-1)

        target_scores = target_scores * norm_align_metric

        return (
            target_labels,
            target_bboxes,
            target_scores,
            fg_mask.bool(),
            target_gt_idx
        )

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate positive sample mask using task-aligned assignment rules.

        Combines three conditions to determine positive anchors:
            1. Anchor is spatially inside the GT bounding box.
            2. Anchor ranks in top-k by task-aligned score for that GT.
            3. GT box is valid (according to mask_gt).

        Args:
            pd_scores (torch.Tensor):
                Predicted class scores, shape (B, H*W, C).
            pd_bboxes (torch.Tensor):
                Predicted bounding boxes, shape (B, H*W, 4).
            gt_labels (torch.Tensor):
                Ground truth class labels, shape (B, N, 1).
            gt_bboxes (torch.Tensor):
                Ground truth boxes, shape (B, N, 4).
            anc_points (torch.Tensor):
                Anchor centers, shape (H*W, 2).
            mask_gt (torch.Tensor):
                GT validity mask, shape (B, N).

        Returns:
            mask_pos (torch.Tensor):
                Positive sample mask, shape (B, N, H*W).
            align_metric (torch.Tensor):
                Task-aligned score matrix, shape (B, N, H*W).
            overlaps (torch.Tensor):
                IoU matrix between GT and anchors, shape (B, N, H*W).
        """
        # Mask of anchors inside GT boxes
        mask_in_gts = self.select_candidates_in_gts(
            anc_points, gt_bboxes)  # (B, N, H*W)

        # Compute task-aligned metric and IoU
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes,
            mask_in_gts * mask_gt
        )

        # Top-k scoring anchors per GT
        mask_topk = self.select_topk_candidates(
            align_metric,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )

        # Final mask: inside GTs ∩ top-k ∩ valid GTs
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute task-aligned metrics and IoU between predicted boxes and ground
        truths.

        The alignment metric is defined as:
            (classification_score ** alpha) * (IoU ** beta)

        Args:
            pd_scores (torch.Tensor):
                Predicted class scores, shape (B, H*W, C).
            pd_bboxes (torch.Tensor):
                Predicted bounding boxes, shape (B, H*W, 4).
            gt_labels (torch.Tensor):
                Ground truth class labels, shape (B, N, 1).
            gt_bboxes (torch.Tensor):
                Ground truth boxes, shape (B, N, 4).
            mask_gt (torch.Tensor):
                Boolean mask indicating valid GTs, shape (B, N, H*W).

        Returns:
            align_metric (torch.Tensor):
                Task-aligned metric, shape (B, N, H*W).
            overlaps (torch.Tensor):
                IoU between each GT and anchor, shape (B, N, H*W).
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # Ensure boolean type

        overlaps = torch.zeros(
            (self.bs, self.n_max_boxes, na),
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device
        )

        bbox_scores = torch.zeros(
            (self.bs, self.n_max_boxes, na),
            dtype=pd_scores.dtype,
            device=pd_scores.device
        )

        # Index: ind[0] = batch index, ind[1] = class index for each GT
        ind = torch.zeros((2, self.bs, self.n_max_boxes), dtype=torch.long)
        ind[0] = torch.arange(self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)  # (B, N)

        # Gather predicted class scores for GT class per anchor
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        # Gather predicted and GT boxes for IoU computation
        pd_boxes = pd_bboxes \
            .unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes \
            .unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]

        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # Compute task-aligned metric
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """
        Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        iou = bbox_iou(gt_bboxes, pd_bboxes, xywh=False, ciou=True)
        iou = iou.squeeze(-1).clamp_(0)
        return iou

    def select_topk_candidates(
        self,
        metrics: torch.Tensor,
        topk_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Select top-k anchors for each GT based on alignment metrics.

        This method produces a binary mask indicating which anchors are
        among the top-k aligned anchors for each GT.

        Args:
            metrics (torch.Tensor):
                Alignment scores between GT and anchors,
                shape (B, N, H*W).
            topk_mask (torch.Tensor | None, optional):
                Optional mask indicating valid GTs. Shape (B, N, topk).
                If None, it is inferred by comparing scores > eps.

        Returns:
            torch.Tensor:
                Binary mask of selected top-k anchors, shape (B, N, H*W).
        """

        # Get top-k values and their indices along anchor dimension
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, dim=-1, largest=True)

        # Filter: only keep top-k entries for valid GTs
        if topk_mask is None:
            # (B, N, topk): keep anchors with at least one metric > eps
            topk_mask = (
                topk_metrics.max(dim=-1, keepdim=True)[0] > self.eps
            ).expand_as(topk_idxs)

        # Invalidate positions with False mask
        topk_idxs.masked_fill_(~topk_mask, 0)

        # Create binary mask over full (B, N, H*W)
        count_tensor = torch.zeros(
            metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(
            topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)

        for k in range(self.topk):
            # scatter 1 at the selected top-k indices
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k: k + 1], ones)

        # Remove duplicate assignments (same anchor assigned multiple times)
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(dtype=metrics.dtype)

    def get_targets(
        self,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate classification labels, bounding boxes, and scores for positive
        anchors.

        Args:
            gt_labels (torch.Tensor):
                Ground truth class labels, shape (B, N, 1).
            gt_bboxes (torch.Tensor):
                Ground truth bounding boxes, shape (B, N, 4).
            target_gt_idx (torch.Tensor):
                Assigned GT indices for each anchor, shape (B, H*W).
            fg_mask (torch.Tensor):
                Foreground mask for each anchor, shape (B, H*W).

        Returns:
            target_labels (torch.Tensor):
                Class labels per anchor, shape (B, H*W).
            target_bboxes (torch.Tensor):
                Bounding boxes per anchor, shape (B, H*W, 4).
            target_scores (torch.Tensor):
                One-hot encoded scores, shape (B, H*W, num_classes).
        """

        # Flatten GTs and shift GT indices per batch
        batch_ind = torch.arange(
            self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes

        # Fetch target labels and boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Clamp to valid range to avoid negative label
        target_labels.clamp_(min=0)

        # One-hot encode class scores: (B, H*W, num_classes)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        # Apply foreground mask
        fg_scores_mask = fg_mask.unsqueeze(-1).expand(-1, -1, self.num_classes)
        target_scores = torch.where(fg_scores_mask.bool(), target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(
        xy_centers: torch.Tensor,
        gt_bboxes: torch.Tensor,
        eps: float = 1e-9
    ) -> torch.Tensor:
        """Determine whether each anchor lies strictly inside each GT bounding box.

        Args:
            xy_centers (torch.Tensor):
                Anchor point centers of shape (H*W, 2).
            gt_bboxes (torch.Tensor):
                Ground truth boxes of shape (B, N, 4), where
                each box is represented as (x1, y1, x2, y2).
            eps (float, optional):
                Small positive value to avoid boundary ambiguity.
                Defaults to 1e-9.

        Returns:
            torch.Tensor:
                Boolean tensor of shape (B, N, H*W), where each element is True
                if the corresponding anchor lies completely inside the GT box.
        """
        num_anchors = xy_centers.shape[0]
        bs, num_boxes, _ = gt_bboxes.shape

        # Split GT boxes into top-left and bottom-right corners
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, dim=2)  # (B*N, 1, 2)

        # Compute deltas: distance from anchor to box borders
        bbox_deltas = torch.cat([
            xy_centers[None] - lt,  # (1, H*W, 2)
            rb - xy_centers[None]   # (1, H*W, 2)
        ], dim=2)  # (B*N, 1, 4) -> cat -> (B*N, H*W, 4)

        # Reshape to (B, N, H*W, 4)
        bbox_deltas = bbox_deltas.view(bs, num_boxes, num_anchors, 4)

        # Return a mask of whether all 4 distances > eps
        return bbox_deltas.amin(dim=3).gt_(eps)  # (B, N, H*W)

    @staticmethod
    def select_highest_overlaps(
        mask_pos: torch.Tensor,
        overlaps: torch.Tensor,
        n_max_boxes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resolve multiple GT assignments by selecting the one with highest IoU.

        When an anchor point is assigned to multiple ground truths (GT),
        only retain the GT with the highest IoU for each anchor.

        Args:
            mask_pos (torch.Tensor):
                Positive mask of shape (B, N, H*W), where B is the batch size,
                N is n_max_boxes, and H*W is the number of anchor points.
            overlaps (torch.Tensor): IoU overlaps of shape (B, N, H*W).
            n_max_boxes (int): Maximum number of GT boxes per image.

        Returns:
            target_gt_idx (torch.Tensor):
                Final assigned GT indices for each anchor, shape (B, H*W).
            fg_mask (torch.Tensor):
                Foreground mask indicating positive anchors, shape (B, H*W).
            mask_pos (torch.Tensor):
                Updated positive mask, shape (B, N, H*W).
        """

        # Initial foreground mask: (B, H*W)
        fg_mask = mask_pos.sum(dim=-2)

        if fg_mask.max() > 1:

            # Case: An anchor is assigned to multiple GTs, (B, N, H*W)
            mask_multi_gts = (
                fg_mask.unsqueeze(1) > 1
            ).expand(-1, n_max_boxes, -1)

            # Get the GT index with highest IoU for each anchor
            max_overlaps_idx = overlaps.argmax(dim=1)  # (B, H*W)

            # Build one-hot mask for maximum IoU GT
            is_max_overlaps = torch.zeros_like(mask_pos)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            # Retain only highest-IoU GT when multiple GTs are assigned
            mask_pos = torch.where(
                mask_multi_gts, is_max_overlaps, mask_pos
            ).float()
            fg_mask = mask_pos.sum(dim=-2)  # Updated foreground mask

        # Final GT index for each anchor (B, H*W)
        target_gt_idx = mask_pos.argmax(dim=-2)

        return target_gt_idx, fg_mask, mask_pos
