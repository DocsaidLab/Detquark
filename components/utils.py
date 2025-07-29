from __future__ import annotations

import copy
import math
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def clone_act(
    default_act: nn.Module,
    spec: Union[bool, nn.Module]  # Python < 3.10 可改用 `Union`
) -> nn.Module:
    """Return an activation module according to *spec*, guaranteeing freshness.

    The helper is useful in configuration-driven code where an activation may be:

    * **Explicit module** - e.g. ``nn.ReLU(inplace=True)``.
      The instance is *deep-copied* to avoid parameter sharing.
    * **Boolean switch** -
      ``True`` → construct a new instance of ``default_act``'s class.
      ``False`` → disable activation by returning :class:`torch.nn.Identity`.

    Args:
        default_act: Canonical activation whose class is used when *spec* is
            ``True`` (the original instance itself is *not* reused).
        spec: Either a concrete :class:`torch.nn.Module` to clone, or a boolean
            flag indicating whether to use the default activation.

    Returns:
        A fresh :class:`torch.nn.Module` matching the semantics of *spec*.

    Raises:
        TypeError: If *spec* is neither ``bool`` nor :class:`torch.nn.Module`.
    """
    if isinstance(spec, nn.Module):
        # Ensure unique parameters/buffers by deep-copying.
        return copy.deepcopy(spec)

    if isinstance(spec, bool):
        return default_act.__class__() if spec else nn.Identity()

    raise TypeError(
        "`spec` must be bool or torch.nn.Module, "
        f"got {type(spec).__name__!s}."
    )


def empty_like(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Return an empty tensor/array with the same shape as *x* and dtype float32.

    Args:
        x: A :class:`torch.Tensor` or :class:`numpy.ndarray` whose shape is used
            to create the new empty tensor/array.

    Returns:
        A new object of the same type as *x* (`torch.Tensor` or
        `numpy.ndarray`) with identical shape but with dtype ``float32`` and
        uninitialized memory.

    Raises:
        TypeError: If *x* is neither a :class:`torch.Tensor` nor a
            :class:`numpy.ndarray`.
    """
    if isinstance(x, torch.Tensor):
        return torch.empty_like(x, dtype=torch.float32)
    if isinstance(x, np.ndarray):
        return np.empty_like(x, dtype=np.float32)

    raise TypeError(
        "`x` must be a torch.Tensor or numpy.ndarray, "
        f"got {type(x).__name__!s}."
    )


def xywh2xyxy(
    x: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Convert bounding boxes from *(x, y, w, h)* to *(x1, y1, x2, y2)*.

    The input represents box centers with width/height; the output gives
    top-left and bottom-right corners.

    Args:
        x: A 2-D or N-D array/tensor whose last dimension is **exactly 4** and
            ordered as ``(x_center, y_center, width, height)``. Supports
            :class:`numpy.ndarray` and :class:`torch.Tensor`.

    Returns:
        A new array/tensor of identical shape containing coordinates in
        ``(x1, y1, x2, y2)`` order and dtype ``float32`` (via
        :func:`empty_like`). The return type matches that of *x*.

    Raises:
        ValueError: If the last dimension of *x* is not size 4.
        TypeError: If *x* is neither :class:`numpy.ndarray` nor
            :class:`torch.Tensor`.
    """
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError(
            "`x` must be a numpy.ndarray or torch.Tensor, "
            f"got {type(x).__name__!s}."
        )
    if x.shape[-1] != 4:
        raise ValueError(
            "Expected last dimension of size 4; "
            f"received shape {tuple(x.shape)!s}."
        )

    # Allocate an uninitialized output of same shape/dtype=float32
    y = empty_like(x)

    # Split into center coordinates and half‐size
    xy = x[..., :2]          # (x_c, y_c)
    wh_half = x[..., 2:] / 2  # (w/2, h/2)

    # Compute corner coordinates
    y[..., :2] = xy - wh_half  # top-left  (x1, y1)
    y[..., 2:] = xy + wh_half  # bottom-right (x2, y2)

    return y


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    eps: float = 1e-6,
    *,
    xywh: bool = False,
    giou: bool = False,
    diou: bool = False,
    ciou: bool = False,
) -> torch.Tensor:
    """Compute IoU and its common variants between two sets of boxes.

    The function accepts broadcastable tensors whose last dimension is 4.
    By default it assumes **corner format** ``(x1, y1, x2, y2)`` to preserve
    backward compatibility with earlier versions.  Set *xywh=True* for
    center-based ``(x, y, w, h)`` input.

    Exactly one of *giou*, *diou*, *ciou* may be ``True``; when none is set the
    plain IoU is returned.

    Args:
        box1: Tensor of shape ``(..., 4)`` containing one or many boxes.
        box2: Tensor broadcastable to ``box1`` and with the same last-dim size.
        eps: Small constant added to denominators to avoid divide-by-zero.
        xywh: If ``True``, inputs are in ``(x, y, w, h)``; otherwise in
            ``(x1, y1, x2, y2)`` (default).
        giou: If ``True``, return Generalised IoU. Mutually exclusive with
            *diou* and *ciou*.
        diou: If ``True``, return Distance IoU.
        ciou: If ``True``, return Complete IoU.

    Returns:
        Tensor with IoU/GIoU/DIoU/CIoU values.  Its shape equals the broadcasted
        shape of the inputs with the last coordinate dimension removed.

    Raises:
        ValueError: If last dimension is not 4 or more than one variant flag
            is set.
    """
    if box1.shape[-1] != 4 or box2.shape[-1] != 4:
        raise ValueError("Inputs must have shape (..., 4).")
    if sum((giou, diou, ciou)) > 1:
        raise ValueError("giou, diou, and ciou are mutually exclusive.")

    # ── Coordinate conversion ────────────────────────────────────────────────
    if xywh:
        # (x, y, w, h) → (x1, y1, x2, y2)
        (x1_c, y1_c, w1, h1), (x2_c, y2_c, w2,
                               h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        half_w1, half_h1 = w1 * 0.5, h1 * 0.5
        half_w2, half_h2 = w2 * 0.5, h2 * 0.5
        b1_x1, b1_y1 = x1_c - half_w1, y1_c - half_h1
        b1_x2, b1_y2 = x1_c + half_w1, y1_c + half_h1
        b2_x1, b2_y1 = x2_c - half_w2, y2_c - half_h2
        b2_x2, b2_y2 = x2_c + half_w2, y2_c + half_h2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    # ── Intersection ────────────────────────────────────────────────────────
    inter_w = torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)
    inter_h = torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)
    inter = inter_w.clamp(min=0) * inter_h.clamp(min=0)

    # ── Union ───────────────────────────────────────────────────────────────
    if xywh:
        area1 = (w1 * h1).clamp(min=0)
        area2 = (w2 * h2).clamp(min=0)
    else:
        w1 = (b1_x2 - b1_x1).clamp(min=0)
        h1 = (b1_y2 - b1_y1).clamp(min=0)
        w2 = (b2_x2 - b2_x1).clamp(min=0)
        h2 = (b2_y2 - b2_y1).clamp(min=0)
        area1, area2 = w1 * h1, w2 * h2

    union = area1 + area2 - inter + eps
    iou = inter / union  # ← plain IoU

    # ── Variants ────────────────────────────────────────────────────────────
    if not any((giou, diou, ciou)):
        return iou  # standard IoU, nothing more to do

    # Enclosing (convex) box dimensions
    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)

    if giou:
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area  # GIoU  1902.09630

    # Center distance squared & convex diagonal squared (for DIoU/CIoU)
    rho2 = (
        ((b1_x1 + b1_x2) - (b2_x1 + b2_x2)).pow(2) +
        ((b1_y1 + b1_y2) - (b2_y1 + b2_y2)).pow(2)
    ) * 0.25
    c2 = cw.pow(2) + ch.pow(2) + eps

    if diou:
        return iou - rho2 / c2  # DIoU 1911.08287

    # -------- CIoU --------
    v = (4 / math.pi ** 2) * (
        torch.atan(w1 / (h1 + eps)) - torch.atan(w2 / (h2 + eps))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)

    return iou - (rho2 / c2 + alpha * v)  # CIoU


def make_anchors(
    feats: Union[Sequence[torch.Tensor], torch.Tensor],
    strides: Sequence[int],
    grid_cell_offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate anchor-center coordinates and their stride scales.

    The function supports two invocation patterns commonly used in object-
    detection code:

    1. **Eager/Train mode**
       `feats` is a sequence of feature maps, each shaped ``(B, C, H, W)``.
       The spatial shapes ``(H, W)`` are taken directly from the tensors.

    2. **Tracing/Export mode**
       `feats` is a tensor of shape ``(N, 2)`` where each row stores
       ``(H, W)`` for a feature level.  This avoids dynamic shape operations
       during e.g. TorchScript or ONNX export.

    For every feature level *i* the routine creates a grid of points located at
    ``(x + offset, y + offset)`` in feature-space pixels, flattens it, and
    concatenates all levels.  A companion tensor stores the stride (pixel size
    in the original image) of each point.

    Args:
        feats: Either a sequence of feature maps *or* a tensor with per-level
            spatial shapes as described above.
        strides: Stride (scale factor) for each feature level.  Must have the
            same length as *feats*.
        grid_cell_offset: Fractional offset that shifts anchors from the
            integer grid origin.  ``0.5`` places points at cell centres.

    Returns:
        Tuple of two tensors ``(points, strides)``:
        * **points** - ``(N, 2)`` float32 tensor of *x-, y-* coordinates.
        * **strides** - ``(N, 1)`` float32 tensor, each row equal to the stride
          corresponding to its point.

    Raises:
        ValueError: If *strides* length differs from *feats* length.
    """
    # Validate inputs ---------------------------------------------------------
    num_levels = len(feats) if isinstance(
        feats, (list, tuple)) else feats.shape[0]
    if num_levels != len(strides):
        raise ValueError(
            "Length of 'strides' must match number of feature levels.")

    anchor_points: list[torch.Tensor] = []
    stride_tensor: list[torch.Tensor] = []

    # Determine dtype / device from first level (works for both invocation modes)
    if isinstance(feats, (list, tuple)):
        dtype, device = feats[0].dtype, feats[0].device
    else:
        dtype, device = feats.dtype, feats.device

    # ------------------------------------------------------------------------
    for i, stride in enumerate(strides):
        if isinstance(feats, (list, tuple)):
            h, w = feats[i].shape[-2:]  # (H, W) from feature map
        else:  # exported / static case: feats[i] holds (H, W)
            h, w = int(feats[i][0]), int(feats[i][1])

        # Coordinate vectors shifted by offset
        sx = torch.arange(w, dtype=dtype, device=device) + grid_cell_offset
        sy = torch.arange(h, dtype=dtype, device=device) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")  # (H, W)

        # Stack to (H*W, 2) and cache
        anchor_points.append(torch.stack((sx, sy), dim=-1).reshape(-1, 2))

        # Corresponding stride for each point, shape (H*W, 1)
        stride_tensor.append(
            torch.full((h * w, 1), float(stride), dtype=dtype, device=device)
        )

    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)


def bbox2dist(
    anchor_points: torch.Tensor,
    bbox: torch.Tensor,
    reg_max: Union[int, float],
) -> torch.Tensor:
    """Convert absolute corner boxes to per-side distance targets.

    The routine receives **anchor point centres** and **bounding boxes in
    corner form** ``(x1, y1, x2, y2)``; it outputs the distances from each
    anchor to the four sides—left, top, right, bottom—clamped to
    ``[0, reg_max)``.  This representation is required by **DFL
    (Distribution-Focal Loss)** style regressors used in modern YOLO/RT-DETR
    heads.

    Args:
        anchor_points: Tensor of shape ``(..., 2)`` containing the
            ``(x, y)`` coordinates of grid centres.
        bbox: Tensor broadcast-compatible with *anchor_points*, last dimension
            must be **4** and ordered ``(x1, y1, x2, y2)``.
        reg_max: Upper bound (exclusive) of the distance range used by the
            discrete distribution (e.g. 16 in YOLOv8).  Accepts ``int`` or
            ``float`` for flexibility.

    Returns:
        Tensor of shape ``(..., 4)`` holding distances ``(l, t, r, b)`` in the
        same dtype/device as the inputs.  Values are clamped to
        ``[0, reg_max - 0.01]`` to stay within the discrete bin range.

    Raises:
        ValueError: If *bbox*'s last dimension is not **4**.
    """
    if bbox.shape[-1] != 4:  # explicit check; avoids silent broadcasting bugs
        raise ValueError(
            "Expected 'bbox' last dimension size 4 (x1, y1, x2, y2).")

    # Split corners and compute distances
    x1y1, x2y2 = bbox.chunk(2, dim=-1)
    distances = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), dim=-1)

    # Clamp to valid range for discrete regression targets
    return distances.clamp_(0, float(reg_max) - 0.01)


def dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    xywh: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """Convert per-side distances back to bounding-box coordinates.

    This is the **inverse operation** of :pyfunc:`bbox2dist`.  It transforms the
    four-tuple distances *(l, t, r, b)*—measured from an anchor-point centre—to
    either centre-based ``(x, y, w, h)`` or corner-based ``(x1, y1, x2, y2)``
    boxes.

    Args:
        distance: Tensor of shape ``(..., 4)`` holding left/top/right/bottom
            distances.  The component dimension is given by *dim*.
        anchor_points: Tensor broadcast-compatible with *distance* whose last
            dimension is **2**, storing the ``(x, y)`` coordinates of grid
            centres.
        xywh: If ``True`` (default), return boxes in ``(x, y, w, h)`` format;
            otherwise return ``(x1, y1, x2, y2)``.
        dim: Dimension index along which the 4 distance values are stored.

    Returns:
        A tensor of bounding boxes in the same dtype/device as the inputs.  The
        output shape equals the broadcasted shape of *distance* and
        *anchor_points*, with the component dimension preserved.

    Raises:
        ValueError: If *distance* does not have size 4 along *dim* or
            *anchor_points*' last dimension is not 2.
    """
    if distance.shape[dim] != 4:
        raise ValueError(
            "`distance` must have 4 elements (l, t, r, b) along the specified dim.")
    if anchor_points.shape[-1] != 2:
        raise ValueError("`anchor_points` last dimension must be 2 (x, y).")

    # Split distances into left-top (lt) and right-bottom (rb) halves.
    lt, rb = distance.chunk(2, dim)

    # Recover corner coordinates.
    x1y1 = anchor_points - lt          # top-left corner
    x2y2 = anchor_points + rb          # bottom-right corner

    if xywh:
        # Centre coordinates and width/height.
        c_xy = (x1y1 + x2y2) * 0.5
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # (x, y, w, h)

    return torch.cat((x1y1, x2y2), dim)     # (x1, y1, x2, y2)


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
