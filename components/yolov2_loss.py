from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn

from .utils import bbox_iou


class YOLOv2Loss(nn.Module):
    """
    Compute YOLOv2 loss: localization regression, objectness, and classification.

    Args:
        anchors (List[Tuple[float, float]]): List of (width, height) anchor box sizes in pixels.
        num_classes (int): Number of object classes.
        img_dim (int): Input image width and height in pixels. Default: 416.
        lambda_coord (float): Weight for coordinate regression loss. Default: 5.0.
        lambda_noobj (float): Weight for no-object confidence loss. Default: 0.5.
        ignore_iou_thr (float): IoU threshold above which negative anchors are ignored. Default: 0.7.
        noobj_iou_thr (float): IoU threshold below which anchors count as negatives. Default: 0.1.
        loss_normalize (Literal["batch", "posneg"]): Normalization mode:
            "batch" divides losses by batch size;
            "posneg" divides coord/cls by positives and noobj by negatives. Default: "batch".
    """

    def __init__(
        self,
        anchors: List[Tuple[float, float]],
        num_classes: int,
        img_dim: int = 416,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        ignore_iou_thr: float = 0.7,
        noobj_iou_thr: float = 0.1,
        loss_normalize: Literal["batch", "posneg"] = "batch",
    ):
        super().__init__()

        # Convert anchor sizes to tensor and register as buffers
        anc = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anc)
        self.register_buffer("anchor_w", anc[:, 0] / img_dim)
        self.register_buffer("anchor_h", anc[:, 1] / img_dim)

        # Save parameters
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.ignore_iou_thr = ignore_iou_thr
        self.noobj_iou_thr = noobj_iou_thr

        # Validate IoU thresholds
        if not (0.0 <= self.noobj_iou_thr <= self.ignore_iou_thr < 1.0):
            raise ValueError(
                "noobj_iou_thr must satisfy 0.0 <= noobj_iou_thr <= ignore_iou_thr < 1.0")

        # Initialize loss functions with sum reduction
        # Using sum reduction for stability before normalization
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(reduction="sum")

        # Validate normalization mode
        if loss_normalize not in ("batch", "posneg"):  # ensure supported modes
            raise ValueError("loss_normalize must be 'batch' or 'posneg'")
        self.loss_normalize = loss_normalize

    def forward(
        self,
        preds: torch.Tensor,  # shape: (B, S, S, A*(5 + C))
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and component losses.

        Args:
            preds (Tensor): Raw model predictions
                of shape (B, S, S, A*(5+C)).
            targets (List[Dict]): Ground truth per image with keys 'boxes' and 'labels'.

        Returns:
            total_loss (Tensor): Sum of all loss terms.
            loss_dict (dict): Detached losses for logging.
        """
        device = preds.device
        B, S, S2, D = preds.shape

        # Ensure spatial dimensions match
        if S != S2:
            raise ValueError("Feature map must be square")

        A = self.anchors.size(0)
        expected_D = A * (5 + self.num_classes)
        if D != expected_D:
            message = f"pred channels={D}, expected={expected_D}"
            raise ValueError(message)

        # Reshape to (B, S, S, A, 5 + C)
        preds = preds.view(B, S, S, A, 5 + self.num_classes)
        tx = preds[..., 0]
        ty = preds[..., 1]
        tw = preds[..., 2]
        th = preds[..., 3]
        to = preds[..., 4]
        tcls = preds[..., 5:]

        # Build target tensors and masks
        (
            tgt_xywh,
            tgt_conf,
            tgt_cls_idx,
            obj_mask,
            noobj_mask,
        ) = self.build_targets_yolov2(
            targets,
            self.anchors,
            S,
            self.img_dim,
            self.num_classes,
            self.ignore_iou_thr,
            self.noobj_iou_thr,
            device,
        )

        # Count positives and negatives, avoid zero division
        n_pos = obj_mask.sum().float().clamp(min=1.0)
        n_neg = noobj_mask.sum().float().clamp(min=1.0)

        # Coordinate loss (only positive anchors)
        if obj_mask.any():
            # Extract predictions and targets at positive positions
            pos_xywh = tgt_xywh[obj_mask]
            pred_x = torch.sigmoid(tx[obj_mask])
            pred_y = torch.sigmoid(ty[obj_mask])
            pred_w = tw[obj_mask]
            pred_h = th[obj_mask]

            loss_coord = self.lambda_coord * (
                self.mse(pred_x, pos_xywh[:, 0])
                + self.mse(pred_y, pos_xywh[:, 1])
                + self.mse(pred_w, pos_xywh[:, 2])
                + self.mse(pred_h, pos_xywh[:, 3])
            )
        else:
            loss_coord = torch.tensor(0.0, device=device)

        # Objectness and no-objectness losses
        if obj_mask.any():
            loss_obj = self.bce(to[obj_mask], tgt_conf[obj_mask])
        else:
            loss_obj = torch.tensor(0.0, device=device)
        if noobj_mask.any():
            loss_noobj = self.bce(to[noobj_mask], tgt_conf[noobj_mask])
        else:
            loss_noobj = torch.tensor(0.0, device=device)
        loss_noobj *= self.lambda_noobj

        # Classification loss (only positive anchors)
        if obj_mask.any():
            cls_logits = tcls[obj_mask].view(-1, self.num_classes)
            cls_tgt = tgt_cls_idx[obj_mask].view(-1)
            loss_cls = self.ce(cls_logits, cls_tgt)
        else:
            loss_cls = torch.tensor(0.0, device=device)

        # Normalize losses
        if self.loss_normalize == "batch":
            div = float(B)
            loss_coord /= div
            loss_obj /= div
            loss_noobj /= div
            loss_cls /= div
        else:
            loss_coord /= n_pos
            loss_obj /= n_pos
            loss_noobj /= n_neg
            loss_cls /= n_pos

        total_loss = loss_coord + loss_obj + loss_noobj + loss_cls

        # Prepare logging dictionary
        loss_dict = {
            "loss": total_loss.detach(),
            "coord": loss_coord.detach(),
            "obj": loss_obj.detach(),
            "noobj": loss_noobj.detach(),
            "cls": loss_cls.detach(),
            "pos": n_pos.detach(),
            "neg": n_neg.detach(),
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def build_targets_yolov2(
        self,
        targets: List[Dict[str, torch.Tensor]],
        anchors: torch.Tensor,
        S: int,
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
        torch.Tensor
    ]:
        """
        Generate training targets and masks for YOLOv2.

        Args:
            targets (List[Dict]): Each dict contains 'boxes' (Tensor[N,4])
                and 'labels' (Tensor[N]).
            anchors (Tensor): Anchor sizes in pixels, shape (A, 2).
            S (int): Grid size per spatial dimension.
            img_dim (int): Image dimension in pixels.
            num_classes (int): Number of object classes.
            ignore_iou_thr (float): IoU threshold to ignore negatives.
            noobj_iou_thr (float): IoU threshold to count as negatives.
            device (torch.device): Device for tensors.

        Returns:
            target_xywh (Tensor): Shape (B, S, S, A, 4) with tx, ty, tw, th.
            target_conf (Tensor): Shape (B, S, S, A) objectness targets.
            target_cls_idx (Tensor): Shape (B, S, S, A) class indices.
            obj_mask (BoolTensor): Positive anchor mask.
            noobj_mask (BoolTensor): Negative anchor mask.
        """
        B = len(targets)
        A = anchors.size(0)

        # Normalize anchor box to grid
        anchor_w = anchors[:, 0] / img_dim
        anchor_h = anchors[:, 1] / img_dim
        anchor_boxes = torch.stack(
            [-anchor_w/2, -anchor_h/2, anchor_w/2, anchor_h/2],
            dim=1
        ).to(device)

        # Allocate output tensors
        target_xywh = torch.zeros((B, S, S, A, 4), device=device)
        target_conf = torch.zeros((B, S, S, A), device=device)
        target_cls_idx = torch.zeros(
            (B, S, S, A), dtype=torch.long, device=device)
        obj_mask = torch.zeros((B, S, S, A), dtype=torch.bool, device=device)
        noobj_mask = torch.ones_like(obj_mask)

        # Populate targets per image
        for b_idx, sample in enumerate(targets):
            boxes = sample.get("boxes", torch.empty((0, 4))).to(device)
            labels = sample.get("labels", torch.empty(
                (0,), dtype=torch.long)).to(device)

            # Skip if no objects
            if boxes.numel() == 0:
                continue

            # Validate class labels
            if labels.max() >= num_classes:
                raise ValueError("label id >= num_classes")

            # Compute centers, sizes
            x1, y1, x2, y2 = boxes.unbind(1)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = (x2 - x1).clamp(min=1e-6)
            h = (y2 - y1).clamp(min=1e-6)

            # Determine grid cell indices
            gi = (cx * S).floor().clamp(0, S - 1).long()
            gj = (cy * S).floor().clamp(0, S - 1).long()

            for k in range(boxes.size(0)):
                i, j = gi[k], gj[k]

                # Compute IoU between gt box and anchors
                gt_box = torch.tensor(
                    [-w[k]/2, -h[k]/2, w[k]/2, h[k]/2],
                    device=device
                ).unsqueeze(0)
                ious = bbox_iou(anchor_boxes, gt_box).view(-1)
                best_a = torch.argmax(ious)

                # Skip if anchor already assigned
                if obj_mask[b_idx, j, i, best_a]:
                    continue

                # Mark positive and clear no-object
                obj_mask[b_idx, j, i, best_a] = True
                noobj_mask[b_idx, j, i, best_a] = False

                # Assign target offsets and sizes
                target_xywh[b_idx, j, i, best_a, 0] = cx[k] * S - i
                target_xywh[b_idx, j, i, best_a, 1] = cy[k] * S - j
                target_xywh[b_idx, j, i, best_a, 2] = torch.log(
                    w[k] / anchor_w[best_a] + 1e-16)
                target_xywh[b_idx, j, i, best_a, 3] = torch.log(
                    h[k] / anchor_h[best_a] + 1e-16)

                # Set objectness and class
                target_conf[b_idx, j, i, best_a] = 1.0
                target_cls_idx[b_idx, j, i, best_a] = labels[k]

                # Suppress no-object in IoU "gray zone"
                mask_high = ious > ignore_iou_thr
                mask_mid = ious > noobj_iou_thr
                suppress_mask = mask_high | mask_mid
                if suppress_mask.any():
                    noobj_mask[b_idx, j, i, suppress_mask] = False

        return target_xywh, target_conf, target_cls_idx, obj_mask, noobj_mask
