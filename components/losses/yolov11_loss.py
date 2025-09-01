from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..assigners import TaskAlignedAssigner
from ..utils import dist2bbox, make_anchors, xywh2xyxy
from .bbox_loss import BBoxLoss


class YOLOv11Loss(nn.Module):
    """
    Composite loss for anchor-free YOLOv11, combining box, classification, and DFL losses.

    This loss module tightly couples with the YOLOv11Head:
    - Expected output channels: 4 * reg_max (box) + num_classes (cls)
    - If using objectness logits, include an additional logit before class scores.
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int,
        strides: Tuple[int, ...],
        box_weight: float = 7.5,
        cls_weight: float = 1.0,
        dfl_weight: float = 1.5,
        tal_topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
    ) -> None:
        """
        Initialize the YOLOv11 loss module.

        Args:
            num_classes: Number of target classes.
            reg_max: Maximum regression value for DFL.
            strides: Stride values for each detection layer (e.g., (8, 16, 32)).
            box_weight: Weight for the IoU-based box loss.
            cls_weight: Weight for the classification loss.
            dfl_weight: Weight for the distribution focal loss.
            tal_topk: Top-k candidates to consider in TaskAlignedAssigner.
            alpha: Alpha parameter for classification component in TAL.
            beta: Beta parameter for localization component in TAL.
        """
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.no = 4 * reg_max + num_classes
        self.register_buffer(
            "stride",
            torch.tensor(strides, dtype=torch.float32)
        )

        # Loss weights
        self.box_gain = box_weight
        self.cls_gain = cls_weight
        self.dfl_gain = dfl_weight

        # Submodules
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BBoxLoss(reg_max)
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
        )

        # DFL projection vector
        proj = torch.arange(reg_max, dtype=torch.float32)
        self.register_buffer("proj", proj)

    @staticmethod
    def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from (x1, y1, x2, y2) to (xc, yc, w, h) format.

        Args:
            boxes: Tensor of shape (N, 4) in x1, y1, x2, y2 format.

        Returns:
            Tensor of shape (N, 4) in x_center, y_center, width, height format.
        """
        x1, y1, x2, y2 = boxes.unbind(-1)
        xc = (x1 + x2) * 0.5
        yc = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return torch.stack([xc, yc, w, h], dim=-1)

    def _listdict_to_targets(
        self,
        targets: List[Dict[str, torch.Tensor]],
        img_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Convert dataloader targets (list of dicts) to a single tensor.

        Args:
            targets: List of dictionaries with keys 'boxes' and 'labels'.
            img_hw: Tuple of (image_height, image_width) in pixels.

        Returns:
            Tensor of shape (T, 6), where each row is
            (image_index, label, x_center, y_center, width, height) in pixel coordinates.
        """
        H, W = img_hw
        rows: List[torch.Tensor] = []
        for img_idx, t in enumerate(targets):
            boxes = t["boxes"]
            dtype, device = boxes.dtype, boxes.device

            # Ensure boxes tensor
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.as_tensor(boxes, dtype=dtype, device=device)
            else:
                boxes = boxes.float().clone()

            # Scale from normalized [0,1] to pixel if necessary
            scale = torch.tensor([W, H, W, H], dtype=dtype, device=device)
            boxes = boxes * scale

            labels = t["labels"]
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(
                    labels, dtype=torch.long, device=device)

            cxcywh = self._xyxy_to_cxcywh(boxes)
            batch_idx = torch.full((cxcywh.size(0), 1), img_idx, device=device)
            label_col = labels.view(-1, 1).float()
            rows.append(torch.cat((batch_idx, label_col, cxcywh), dim=1))

        if rows:
            return torch.cat(rows, dim=0)

        # If No targets
        return torch.zeros(0, 6, dtype=dtype, device=device)

    def _preprocess_targets(
        self,
        targets: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Pad and reshape targets for batch processing.

        Args:
            targets: Tensor of shape (T, 6) from _listdict_to_targets.
            batch_size: Number of images in the batch.

        Returns:
            Tensor of shape (batch_size, max_objs, 5), where each row is
            (label, x1, y1, x2, y2) in pixel coordinates, padded with zeros.
        """
        nl, ne = targets.shape
        device = targets.device
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=device)

        img_indices = targets[:, 0].long()

        # 計算每張圖內有幾個 box
        counts = torch.bincount(img_indices, minlength=batch_size)

        max_objs = int(counts.max())

        # ne - 1 是因為扣除原本的 image index
        padded = torch.zeros(batch_size, max_objs, ne - 1, device=device)

        for img_id in range(batch_size):
            rows = targets[img_indices == img_id, 1:]
            padded[img_id, : rows.size(0)] = rows

        # Convert center format to xyxy
        padded[..., 1:5] = xywh2xyxy(padded[..., 1:5])

        return padded

    def _bbox_decode(
        self,
        anchor_points: torch.Tensor,
        pred_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode discrete DFL distribution to bounding boxes in xyxy format.

        Args:
            anchor_points: Tensor of shape (B, A, 2) with anchor locations.
            pred_dist: Tensor of shape (B, A, 4*R) of logits or distances.

        Returns:
            Tensor of shape (B, A, 4) in xyxy coordinates.
        """
        if self.reg_max > 1:
            b, a, ch = pred_dist.shape
            prob = pred_dist.view(b, a, 4, ch // 4).softmax(-1)
            pred_dist = (prob * self.proj.to(prob.dtype)).sum(-1)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(
        self,
        preds: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and logging dictionary for YOLOv11.

        Args:
            preds:
                List of prediction tensors from each detection layer,
                each of shape (B, no, H, W).
            targets:
                Dataloader targets as a list of dicts with 'boxes' and 'labels'.

        Returns:
            total_loss: Scalar tensor of combined losses.
            log_dict:
                Dictionary with individual loss components:
                {'box': box_loss, 'cls': cls_loss, 'dfl': dfl_loss}.
        """

        # Rearrange head outputs to (B, A, *)
        merged = torch.cat([
            x.view(preds[0].shape[0], self.no, -1) for x in preds], dim=2)
        pred_distri, pred_scores = \
            torch.split(merged, (4 * self.reg_max, self.nc), dim=1)

        # Permute for processing
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        device = pred_scores.device
        batch_size = pred_scores.shape[0]

        # Generate anchors and strides
        anchor_points, stride_tensor = make_anchors(preds, self.stride, 0.5)

        # Convert GT from list[dict] -> padded tensor
        img_h = preds[0].shape[2] * self.stride[0]
        img_w = preds[0].shape[3] * self.stride[0]

        # targets: [B,] -> [B * objs, 6(idx, lb, cx, cy, w, h)]
        targets_tensor = self._listdict_to_targets(targets, (img_h, img_w))
        targets_pad = self._preprocess_targets(targets_tensor, batch_size)
        gt_labels, gt_bboxes = targets_pad.split((1, 4), dim=2)

        # 很多 box 是 padding 的，必須把它們標記出來
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Decode bboxes and assign with TAL
        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type_as(gt_bboxes),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1.0)

        # Classification loss (BCE)
        loss_cls = self.bce(
            pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Box IoU and DFL loss for positive anchors
        loss_box = torch.zeros((), device=device)
        loss_dfl = torch.zeros((), device=device)
        if fg_mask.any():
            tgt_boxes_fs = target_bboxes / stride_tensor
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                tgt_boxes_fs,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        # Weighted sum and return
        total_loss = (
            loss_box * self.box_gain +
            loss_cls * self.cls_gain +
            loss_dfl * self.dfl_gain
        ) * batch_size

        log_dict = {
            "box": (loss_box * self.box_gain).detach(),
            "cls": (loss_cls * self.cls_gain).detach(),
            "dfl": (loss_dfl * self.dfl_gain).detach(),
        }

        return total_loss, log_dict
