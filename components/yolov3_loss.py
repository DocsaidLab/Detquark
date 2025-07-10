from typing import Dict, List, Literal, Sequence, Tuple

import torch
import torch.nn as nn

from .utils import bbox_iou


class YOLOv3Loss(nn.Module):
    """
    Multi-scale YOLOv3 loss combining bounding box regression, objectness,
    and per-class binary cross-entropy classification.

    Args:
        anchors (Sequence[Tuple[float, float]]): Global anchor list ``[(w, h), ...]``
            in pixels with total A anchors.
        anchor_masks (Sequence[Sequence[int]]): Per-scale anchor indices, e.g.
            ``[[6,7,8], [3,4,5], [0,1,2]]``.
        strides (Sequence[int]): Down-sampling factor per scale, e.g. ``[32, 16, 8]``.
        num_classes (int): Number of object classes *C*.
        img_dim (int, optional): Nominal input image size in pixels, used for
            anchor normalization. Default is 416.
        lambda_coord (float, optional): Weight for coordinate regression loss.
            Default is 5.0.
        lambda_noobj (float, optional): Weight for *no-object* BCE loss.
            Default is 0.5.
        ignore_iou_thr (float, optional): IoU threshold above which no-object
            loss is ignored (gray zone). Default is 0.7.
        noobj_iou_thr (float, optional): IoU threshold below which no-object
            loss is fully counted. Default is 0.1.
        loss_normalize (Literal["batch", "posneg"], optional): Normalization mode
            for losses, either "batch" or "posneg". Default is "batch".
    """

    def __init__(
        self,
        anchors: Sequence[Tuple[float, float]],
        anchor_masks: Sequence[Sequence[int]],
        strides: Sequence[int],
        num_classes: int,
        img_dim: int = 416,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        ignore_iou_thr: float = 0.7,
        noobj_iou_thr: float = 0.1,
        loss_normalize: Literal["batch", "posneg"] = "batch",
    ) -> None:
        super().__init__()

        if len(anchor_masks) != len(strides):
            raise ValueError(
                "`anchor_masks` and `strides` must have the same length")

        if not (0.0 <= noobj_iou_thr <= ignore_iou_thr < 1.0):
            raise ValueError("Require 0 ≤ noobj_iou_thr ≤ ignore_iou_thr < 1")

        # Register anchor buffers and initialize hyperparameters
        anc = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anc)  # shape: [A, 2]

        self.anchor_masks = [list(m)
                             for m in anchor_masks]  # deep copy for safety
        self.strides = list(strides)
        self.num_classes = int(num_classes)
        self.img_dim = int(img_dim)

        self.lambda_coord = float(lambda_coord)
        self.lambda_noobj = float(lambda_noobj)
        self.ignore_iou_thr = float(ignore_iou_thr)
        self.noobj_iou_thr = float(noobj_iou_thr)
        self.loss_normalize = loss_normalize

        # Loss functions with sum reduction for stable accumulation
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        preds: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and component losses for multi-scale predictions.

        Args:
            preds (List[torch.Tensor]): List of prediction tensors per scale,
                each with shape ``(B, Sy, Sx, A*(5+C))``.
            targets (List[Dict[str, torch.Tensor]]): Batch ground-truth list
                (length B), each dict with keys:
                - "boxes": Tensor[N, 4] (x1, y1, x2, y2)
                - "labels": Tensor[N] class indices

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - total_loss: scalar tensor representing the total combined loss.
                - loss_dict: dict of detached loss components for logging, keys:
                  ``"loss"``, ``"coord"``, ``"obj"``, ``"noobj"``, ``"cls"``,
                  ``"pos"``, and ``"neg"``.
        """
        if len(preds) != len(self.anchor_masks):
            raise ValueError(
                "Number of prediction tensors must match number of scales")

        B = preds[0].shape[0]
        device = preds[0].device
        dtype = preds[0].dtype

        # Validate shapes and compute grid sizes per scale
        grid_shapes: List[Tuple[int, int]] = []
        for p, mask in zip(preds, self.anchor_masks):
            _, Sy, Sx, D = p.shape
            expected_D = len(mask) * (5 + self.num_classes)
            if D != expected_D:
                raise ValueError(
                    f"Prediction channel dim {D} does not match expected {expected_D} for mask {mask}"
                )
            grid_shapes.append((Sy, Sx))

        # Construct training targets and masks for each scale
        (
            tgt_xywh,
            tgt_obj,
            tgt_cls,
            obj_mask,
            noobj_mask,
        ) = self.build_targets_yolov3(
            targets,
            self.anchors,
            self.anchor_masks,
            self.strides,
            grid_shapes,
            self.img_dim,
            self.num_classes,
            self.ignore_iou_thr,
            self.noobj_iou_thr,
            device,
        )

        # Initialize accumulators for losses and positive/negative counts
        loss_coord = torch.tensor(0.0, device=device, dtype=dtype)
        loss_obj = torch.tensor(0.0, device=device, dtype=dtype)
        loss_noobj = torch.tensor(0.0, device=device, dtype=dtype)
        loss_cls = torch.tensor(0.0, device=device, dtype=dtype)
        n_pos, n_neg = 0.0, 0.0

        # Iterate over scales to compute component losses
        for p_raw, t_xywh, t_obj, t_cls, pos_m, neg_m in zip(
            preds, tgt_xywh, tgt_obj, tgt_cls, obj_mask, noobj_mask
        ):
            B_, Sy, Sx, _ = p_raw.shape
            A = t_obj.shape[-1]
            p = p_raw.view(B_, Sy, Sx, A, 5 + self.num_classes)

            # Coordinate regression loss on positive anchors
            if pos_m.any():
                p_x = torch.sigmoid(p[..., 0])[pos_m]
                p_y = torch.sigmoid(p[..., 1])[pos_m]
                p_w = p[..., 2][pos_m]
                p_h = p[..., 3][pos_m]

                t_x = t_xywh[..., 0][pos_m]
                t_y = t_xywh[..., 1][pos_m]
                t_w = t_xywh[..., 2][pos_m]
                t_h = t_xywh[..., 3][pos_m]

                loss_coord += self.lambda_coord * (
                    self.mse(p_x, t_x)
                    + self.mse(p_y, t_y)
                    + self.mse(p_w, t_w)
                    + self.mse(p_h, t_h)
                )

            # Objectness loss for positive and negative samples
            if pos_m.any():
                loss_obj += self.bce(p[..., 4][pos_m], t_obj[pos_m])
            if neg_m.any():
                loss_noobj += self.bce(p[..., 4][neg_m], t_obj[neg_m])

            # Classification loss on positive samples (multi-label BCE)
            if pos_m.any():
                loss_cls += self.bce(p[..., 5:][pos_m], t_cls[pos_m])

            # Count positive and negative anchors
            n_pos += pos_m.sum().item()
            n_neg += neg_m.sum().item()

        # Scale no-object loss by its lambda weight
        loss_noobj *= self.lambda_noobj

        # Normalize losses by batch size or positive/negative counts
        if self.loss_normalize == "batch":
            div_pos = float(B)
            div_neg = float(B)
        else:  # "posneg"
            div_pos = max(n_pos, 1.0)
            div_neg = max(n_neg, 1.0)

        loss_coord /= div_pos
        loss_obj /= div_pos
        loss_cls /= div_pos
        loss_noobj /= div_neg

        total_loss = loss_coord + loss_obj + loss_cls + loss_noobj

        return total_loss, {
            "loss": total_loss.detach(),
            "coord": loss_coord.detach(),
            "obj": loss_obj.detach(),
            "noobj": loss_noobj.detach(),
            "cls": loss_cls.detach(),
            "pos": torch.tensor(n_pos, device=device),
            "neg": torch.tensor(n_neg, device=device),
        }

    @torch.no_grad()
    def build_targets_yolov3(
        self,
        targets: List[Dict[str, torch.Tensor]],
        anchors: torch.Tensor,
        anchor_masks: Sequence[Sequence[int]],
        strides: Sequence[int],
        grid_shapes: Sequence[Tuple[int, int]],
        img_dim: int,
        num_classes: int,
        ignore_iou_thr: float,
        noobj_iou_thr: float,
        device: torch.device,
    ) -> Tuple[
        List[torch.Tensor],  # tgt_xywh
        List[torch.Tensor],  # tgt_obj
        List[torch.Tensor],  # tgt_cls
        List[torch.Tensor],  # obj_mask
        List[torch.Tensor],  # noobj_mask
    ]:
        """
        Generate multi-scale training targets for YOLOv3.

        Args:
            targets (List[Dict[str, torch.Tensor]]): Batch ground truth with keys:
                - "boxes": Tensor[N, 4], bounding boxes (x1, y1, x2, y2).
                - "labels": Tensor[N], class indices.
            anchors (torch.Tensor): All anchors tensor, shape [A, 2].
            anchor_masks (Sequence[Sequence[int]]): Anchor index groups per scale.
            strides (Sequence[int]): Down-sampling strides per scale.
            grid_shapes (Sequence[Tuple[int, int]]): Feature map spatial sizes per scale.
            img_dim (int): Nominal input image size in pixels.
            num_classes (int): Number of object classes.
            ignore_iou_thr (float): IoU threshold above which no-object loss is ignored.
            noobj_iou_thr (float): IoU threshold below which no-object loss is counted.
            device (torch.device): Device to allocate tensors on.

        Returns:
            Tuple of lists of tensors per scale:
            - tgt_xywh: target bounding box offsets ``(B, Sy, Sx, A, 4)``
            - tgt_obj: target objectness ``(B, Sy, Sx, A)``
            - tgt_cls: target class one-hot encoding ``(B, Sy, Sx, A, C)``
            - obj_mask: positive sample boolean mask ``(B, Sy, Sx, A)``
            - noobj_mask: negative sample boolean mask ``(B, Sy, Sx, A)``
        """
        B = len(targets)
        num_scales = len(anchor_masks)

        # Allocate zero tensors for targets and masks per scale
        tgt_xywh, tgt_obj, tgt_cls = [], [], []
        obj_mask, noobj_mask = [], []

        for (Sy, Sx), mask in zip(grid_shapes, anchor_masks):
            A = len(mask)
            tgt_xywh.append(torch.zeros((B, Sy, Sx, A, 4), device=device))
            tgt_obj.append(torch.zeros((B, Sy, Sx, A), device=device))
            tgt_cls.append(torch.zeros(
                (B, Sy, Sx, A, num_classes), device=device))

            obj_mask.append(torch.zeros(
                (B, Sy, Sx, A), dtype=torch.bool, device=device))
            noobj_mask.append(torch.ones(
                (B, Sy, Sx, A), dtype=torch.bool, device=device))

        # Prepare normalized anchor boxes centered at origin for IoU calc
        anc_w = anchors[:, 0] / img_dim
        anc_h = anchors[:, 1] / img_dim
        anchor_boxes = torch.stack(
            (-anc_w / 2, -anc_h / 2, anc_w / 2, anc_h / 2), dim=1
        ).to(device)

        # Process each image in the batch
        for b, sample in enumerate(targets):
            boxes = sample["boxes"].to(device)
            labels = sample["labels"].to(device)

            if boxes.numel() == 0:
                continue

            # Convert normalized boxes [0,1] to pixel coords if needed
            if boxes.max() <= 1.0:
                boxes = boxes * img_dim

            x1, y1, x2, y2 = boxes.unbind(1)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            w, h = (x2 - x1).clamp(min=1e-6), (y2 - y1).clamp(min=1e-6)

            for k in range(boxes.size(0)):
                # 1. Find best matching anchor for current GT box
                gt_box_norm = torch.tensor(
                    [-w[k] / (2 * img_dim), -h[k] / (2 * img_dim),
                     w[k] / (2 * img_dim),  h[k] / (2 * img_dim)],
                    device=device,
                ).unsqueeze(0)

                # IoUs with all anchors [A]
                ious = bbox_iou(anchor_boxes, gt_box_norm).view(-1)
                best_a = int(ious.argmax())

                # Locate scale index and local anchor index within that scale
                for scale_idx, mask in enumerate(anchor_masks):
                    if best_a in mask:
                        s = scale_idx
                        a_local = mask.index(best_a)
                        break

                Sy, Sx = grid_shapes[s]
                stride = strides[s]

                # Compute grid cell indices for GT center
                cx_cell, cy_cell = cx[k] / stride, cy[k] / stride
                gi = int(torch.floor(cx_cell).clamp(0, Sx - 1))
                gj = int(torch.floor(cy_cell).clamp(0, Sy - 1))

                # Skip if anchor cell already assigned positive
                if obj_mask[s][b, gj, gi, a_local]:
                    continue

                # 2. Assign positive samples
                obj_mask[s][b, gj, gi, a_local] = True
                noobj_mask[s][b, gj, gi, a_local] = False

                tgt_xywh[s][b, gj, gi, a_local, 0] = cx_cell - gi
                tgt_xywh[s][b, gj, gi, a_local, 1] = cy_cell - gj
                tgt_xywh[s][b, gj, gi, a_local, 2] = torch.log(
                    w[k] / anchors[best_a, 0] + 1e-16)
                tgt_xywh[s][b, gj, gi, a_local, 3] = torch.log(
                    h[k] / anchors[best_a, 1] + 1e-16)

                tgt_obj[s][b, gj, gi, a_local] = 1.0
                tgt_cls[s][b, gj, gi, a_local, labels[k]] = 1.0

                # 3. Suppress no-object loss (gray zone) for anchors with intermediate IoU
                for a_id, iou in enumerate(ious):
                    for s2, mask2 in enumerate(anchor_masks):
                        if a_id not in mask2:
                            continue
                        a_local2 = mask2.index(a_id)

                        Sy2, Sx2 = grid_shapes[s2]
                        stride2 = strides[s2]
                        gi2 = int(torch.floor(
                            cx[k] / stride2).clamp(0, Sx2 - 1))
                        gj2 = int(torch.floor(
                            cy[k] / stride2).clamp(0, Sy2 - 1))

                        if iou >= ignore_iou_thr or iou > noobj_iou_thr:
                            noobj_mask[s2][b, gj2, gi2, a_local2] = False

        return tgt_xywh, tgt_obj, tgt_cls, obj_mask, noobj_mask
