from typing import Dict, List, Tuple

import torch


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute Intersection-over-Union (IoU) between two sets of axis-aligned boxes.

    Args:
        box1 (Tensor[..., 4]): Boxes in [x1, y1, x2, y2] format.
        box2 (Tensor[..., 4]): Boxes in [x1, y1, x2, y2] format.
        eps (float): Small constant to avoid division by zero. Default is 1e-6.

    Returns:
        Tensor[...,]: IoU values with broadcasted shape.
    """
    # ------------------------------------------------------------------
    # 1. Compute intersection coordinates
    # ------------------------------------------------------------------
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    # ------------------------------------------------------------------
    # 2. Compute intersection area
    # ------------------------------------------------------------------
    inter_w = (inter_x2 - inter_x1).clamp(min=0)  # Prevent negative width
    inter_h = (inter_y2 - inter_y1).clamp(min=0)  # Prevent negative height
    inter_area = inter_w * inter_h

    # ------------------------------------------------------------------
    # 3. Compute individual box areas
    # ------------------------------------------------------------------
    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * \
            (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * \
            (box2[..., 3] - box2[..., 1]).clamp(min=0)

    # ------------------------------------------------------------------
    # 4. Compute IoU
    # ------------------------------------------------------------------
    return inter_area / (area1 + area2 - inter_area + eps)


@torch.no_grad()
def build_targets_yolov2(
    targets: List[Dict[str, torch.Tensor]],
    anchors: torch.Tensor,           # shape (A, 2) in pixels
    S: int,
    img_dim: int,
    num_classes: int,
    ignore_iou_thr: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training targets for YOLOv2 detector.

    Args:
        targets (List[Dict[str, Tensor]]): Ground-truth list, each dict contains:
            - 'boxes' (N, 4): normalized xyxy coords
            - 'labels' (N,): class indices
        anchors (Tensor): Anchor sizes (A, 2) in pixels.
        S (int): Number of grid cells per side.
        img_dim (int): Input image dimension (pixels).
        num_classes (int): Total number of classes.
        ignore_iou_thr (float): IoU threshold to ignore no-object loss. Default 0.7.
        device (torch.device): Device for output tensors. Default CPU.

    Returns:
        Tuple containing:
        - target_xywh (Tensor[B, S, S, A, 4]): encoded tx, ty, tw, th
        - target_conf (Tensor[B, S, S, A]): objectness targets (0 or 1)
        - target_cls_idx (Tensor[B, S, S, A]): class indices or -1
        - obj_mask (BoolTensor[B, S, S, A]): mask for positive anchors
        - noobj_mask (BoolTensor[B, S, S, A]): mask for negative anchors

    Raises:
        AssertionError: If S <= 0, img_dim <= 0, or anchors not shape (A, 2).
    """
    # Validate inputs
    assert S > 0 and img_dim > 0, "S and img_dim must be positive integers"
    assert anchors.ndim == 2 and anchors.size(
        1) == 2, "anchors must have shape (A,2)"

    B = len(targets)
    A = anchors.size(0)

    # ------------------------------------------------------------------
    # Precompute normalized anchor boxes centered at origin
    # ------------------------------------------------------------------
    anchor_w = anchors[:, 0] / img_dim  # (A,)
    anchor_h = anchors[:, 1] / img_dim  # (A,)
    anchor_boxes = torch.stack([
        -anchor_w / 2,  # x1
        -anchor_h / 2,  # y1
        anchor_w / 2,  # x2
        anchor_h / 2,  # y2
    ], dim=1).to(device)                  # (A, 4)

    # ------------------------------------------------------------------
    # Allocate target tensors
    # ------------------------------------------------------------------
    target_xywh = torch.zeros((B, S, S, A, 4), device=device)
    target_conf = torch.zeros((B, S, S, A), device=device)
    target_cls_idx = torch.full(
        (B, S, S, A), -1, dtype=torch.long, device=device)
    obj_mask = torch.zeros((B, S, S, A), dtype=torch.bool, device=device)
    noobj_mask = torch.ones_like(obj_mask)

    # ------------------------------------------------------------------
    # Encode each sample's ground-truth boxes
    # ------------------------------------------------------------------
    for b_idx, sample in enumerate(targets):
        boxes = sample.get("boxes", torch.empty(0, 4))
        labels = sample.get("labels", torch.empty(0, dtype=torch.long))
        if boxes.numel() == 0:
            continue

        boxes = boxes.to(device)
        labels = labels.to(device)

        # Ensure class indices valid
        if labels.numel() > 0:
            assert labels.max().item() < num_classes, (
                f"Label index {labels.max().item()} >= num_classes {num_classes}"
            )

        # Convert boxes to center format (cx, cy, w, h)
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)

        # Determine grid cell indices
        grid_x = (cx * S).floor().clamp(0, S - 1).long()
        grid_y = (cy * S).floor().clamp(0, S - 1).long()

        for i in range(boxes.size(0)):
            gi, gj = grid_x[i], grid_y[i]

            # Compute IoU between this GT shape and all anchors
            gt_shape = torch.tensor([
                -w[i] / 2, -h[i] / 2,
                w[i] / 2,  h[i] / 2
            ], dtype=boxes.dtype, device=device).unsqueeze(0)
            ious = bbox_iou(anchor_boxes, gt_shape).view(-1)
            best_a = torch.argmax(ious)

            # Mark this anchor as positive
            obj_mask[b_idx, gj, gi, best_a] = True
            noobj_mask[b_idx, gj, gi, best_a] = False

            # Encode offsets for box regression
            target_xywh[b_idx, gj, gi, best_a, 0] = cx[i] * S - gi  # tx
            target_xywh[b_idx, gj, gi, best_a, 1] = cy[i] * S - gj  # ty
            target_xywh[b_idx, gj, gi, best_a, 2] = torch.log(
                w[i] / anchor_w[best_a] + 1e-16)                     # tw
            target_xywh[b_idx, gj, gi, best_a, 3] = torch.log(
                h[i] / anchor_h[best_a] + 1e-16)                     # th

            target_conf[b_idx, gj, gi, best_a] = 1.0
            target_cls_idx[b_idx, gj, gi, best_a] = labels[i]

            # Suppress no-object loss for other anchors with high IoU
            ignore_mask = (ious > ignore_iou_thr) & (
                torch.arange(A, device=device) != best_a)
            if ignore_mask.any():
                noobj_mask[b_idx, gj, gi, ignore_mask] = False

    return target_xywh, target_conf, target_cls_idx, obj_mask, noobj_mask
