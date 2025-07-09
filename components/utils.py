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
