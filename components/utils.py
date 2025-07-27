import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _clone_act(default_act: nn.Module, spec: bool | nn.Module) -> nn.Module:
    """Return an activation module per spec, ensuring unique instances."""
    if isinstance(spec, nn.Module):
        return copy.deepcopy(spec)
    if spec:
        return default_act.__class__()  # type: ignore[call-arg]
    return nn.Identity()


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

    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)  # Prevent negative width
    inter_h = (inter_y2 - inter_y1).clamp(min=0)  # Prevent negative height
    inter_area = inter_w * inter_h

    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * \
            (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * \
            (box2[..., 3] - box2[..., 1]).clamp(min=0)

    return inter_area / (area1 + area2 - inter_area + eps)


@torch.no_grad()
def build_targets(
    targets: List[Dict[str, torch.Tensor]],
    anchors: torch.Tensor,
    grid_size: int,
    img_dim: int,
    num_classes: int,
    ignore_iou_thr: float,
    noobj_iou_thr: float,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Construct target tensors and masks for training.

    Args:
        targets: List of ground truth annotations per batch item.
        anchors: Anchor box sizes (A, 2).
        grid_size: Grid size.
        img_dim: Image dimension in pixels.
        num_classes: Number of object classes.
        ignore_iou_thr: IoU threshold to ignore no-object loss.
        noobj_iou_thr: IoU threshold to suppress no-object loss.
        device: Device for target tensors.

    Returns:
        target_xywh: Tensor of shape (B, grid_size, grid_size, A, 4) with target box offsets.
        target_conf: Objectness target tensor (B, grid_size, grid_size, A).
        target_cls_idx: Class index targets (B, grid_size, grid_size, A).
        obj_mask: Boolean mask for positive anchors.
        noobj_mask: Boolean mask for negative anchors.
    """
    B = len(targets)
    A = anchors.size(0)

    anchor_w = anchors[:, 0] / img_dim
    anchor_h = anchors[:, 1] / img_dim
    anchor_boxes = torch.stack(
        [-anchor_w / 2, -anchor_h / 2, anchor_w / 2, anchor_h / 2],
        dim=1
    ).to(device)

    target_xywh = torch.zeros((B, grid_size, grid_size, A, 4), device=device)
    target_conf = torch.zeros((B, grid_size, grid_size, A), device=device)
    target_cls_idx = torch.zeros(
        (B, grid_size, grid_size, A), dtype=torch.long, device=device)
    obj_mask = torch.zeros(
        (B, grid_size, grid_size, A), dtype=torch.bool, device=device)
    noobj_mask = torch.ones_like(obj_mask)

    for b_idx, sample in enumerate(targets):
        boxes = sample.get(
            "boxes", torch.empty((0, 4), device=device))
        labels = sample.get(
            "labels", torch.empty((0,), dtype=torch.long, device=device))

        if boxes.numel() == 0:
            continue

        if labels.max() >= num_classes:
            raise ValueError("label id >= num_classes")

        x1, y1, x2, y2 = boxes.unbind(1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)

        gi = (cx * grid_size).floor().clamp(0, grid_size - 1).long()
        gj = (cy * grid_size).floor().clamp(0, grid_size - 1).long()

        for k in range(boxes.size(0)):
            i, j = gi[k], gj[k]

            gt_box = boxes.new_tensor(
                [-w[k] / 2, -h[k] / 2, w[k] / 2, h[k] / 2],
                device=device
            ).unsqueeze(0)
            ious = bbox_iou(anchor_boxes, gt_box).view(-1)
            best_a = torch.argmax(ious)

            if obj_mask[b_idx, j, i, best_a]:
                continue

            obj_mask[b_idx, j, i, best_a] = True
            noobj_mask[b_idx, j, i, best_a] = False

            target_xywh[b_idx, j, i, best_a, 0] = cx[k] * grid_size - i
            target_xywh[b_idx, j, i, best_a, 1] = cy[k] * grid_size - j
            target_xywh[b_idx, j, i, best_a, 2] = torch.log(
                w[k] / anchor_w[best_a] + 1e-16)
            target_xywh[b_idx, j, i, best_a, 3] = torch.log(
                h[k] / anchor_h[best_a] + 1e-16)

            target_conf[b_idx, j, i, best_a] = 1.0
            target_cls_idx[b_idx, j, i, best_a] = labels[k]

            mask_high = ious > ignore_iou_thr
            mask_mid = ious > noobj_iou_thr
            suppress_mask = mask_high | mask_mid
            if suppress_mask.any():
                noobj_mask[b_idx, j, i, suppress_mask] = False

    return target_xywh, target_conf, target_cls_idx, obj_mask, noobj_mask
