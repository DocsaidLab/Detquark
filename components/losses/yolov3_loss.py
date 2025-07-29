from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn

from ..utils import build_targets


class YOLOv3Loss(nn.Module):
    """Multi-scale loss for YOLOv3.

    Attributes:
        anchors (Tensor): Global anchors in pixels, shape (A, 2).
        anchor_masks (List[Tensor]): List of anchor index subsets per scale.
        strides (List[int]): Downsampling factors per scale.
        num_classes (int): Number of object classes.
        img_dim (int): Input image dimension in pixels.
        lambda_coord (float): Weight for coordinate regression loss.
        lambda_noobj (float): Weight for no-objectness loss.
        ignore_iou_thr (float): IoU threshold to ignore no-object loss.
        noobj_iou_thr (float): IoU threshold to suppress no-object loss.
        loss_normalize (Literal["batch","posneg"]): Normalization mode.
        mse (nn.MSELoss): Mean squared error loss with sum reduction.
        bce_obj (nn.BCEWithLogitsLoss): BCE loss for objectness with sum reduction.
        bce_cls (nn.BCEWithLogitsLoss): BCE loss for multi-label classification with sum reduction.
    """

    def __init__(
        self,
        anchors: List[Tuple[float, float]],
        anchor_masks: List[List[int]],
        strides: List[int],
        num_classes: int,
        img_dim: int = 416,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        ignore_iou_thr: float = 0.7,
        noobj_iou_thr: float = 0.1,
        loss_normalize: Literal["batch", "posneg"] = "batch",
    ) -> None:
        """Initializes YOLOv3 multi-scale loss module.

        Args:
            anchors: List of all anchor box sizes in pixels.
            anchor_masks: List of anchor indices per scale.
            strides: List of downsampling strides per scale.
            num_classes: Number of object classes.
            img_dim: Input image dimension in pixels.
            lambda_coord: Weight for coordinate regression loss.
            lambda_noobj: Weight for no-objectness loss.
            ignore_iou_thr: IoU threshold above which no-object loss is ignored.
            noobj_iou_thr: IoU threshold below which no-object loss is active.
            loss_normalize: Loss normalization mode ("batch" or "posneg").
        """
        super().__init__()

        if len(anchor_masks) != len(strides):
            raise ValueError(
                "`anchor_masks` and `strides` must have the same length")
        if not (0.0 <= noobj_iou_thr <= ignore_iou_thr < 1.0):
            raise ValueError("Require 0 ≤ noobj_iou_thr ≤ ignore_iou_thr < 1")

        # Global anchors in pixel units
        anc_pix = torch.as_tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anc_pix)  # shape [A, 2]

        # Anchor subsets per scale
        self.anchor_masks = [torch.as_tensor(
            m, dtype=torch.long) for m in anchor_masks]
        self.strides = list(strides)

        # Hyperparameters
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.ignore_iou_thr = ignore_iou_thr
        self.noobj_iou_thr = noobj_iou_thr

        if loss_normalize not in ("batch", "posneg"):
            raise ValueError("loss_normalize must be 'batch' or 'posneg'")
        self.loss_normalize = loss_normalize

        # Loss functions with sum reduction
        self.mse = nn.MSELoss(reduction="sum")
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="sum")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="sum")

    def _forward_single_scale(
        self,
        preds: torch.Tensor,                      # (B, S, S, A*(5+C))
        targets: List[Dict[str, torch.Tensor]],
        anchors: torch.Tensor,                # (A, 2) in pixel units
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss and stats for a single scale.

        Args:
            preds: Raw predictions tensor for this scale (B, S, S, A*(5+C)).
            targets: Ground truth annotations for the batch.
            anchors_pix: Anchor boxes for this scale in pixel units (A, 2).
            stride: Downsampling stride for this scale.

        Returns:
            total_loss: Scalar tensor of total loss for this scale.
            stat: Dictionary of individual loss components and counts.
        """
        device = preds.device
        B, Sy, Sx, D = preds.shape

        if Sy != Sx:
            raise ValueError("Feature map must be square (Sy == Sx)")

        A = anchors.size(0)
        expected_D = A * (5 + self.num_classes)
        if D != expected_D:
            raise ValueError(f"Expected last dimension {expected_D}, got {D}")

        # Reshape predictions: (B, Sy, Sx, A, 5+C)
        preds = preds.view(B, Sy, Sx, A, 5 + self.num_classes)
        tx, ty, tw, th, to, tcls = (
            preds[..., 0],
            preds[..., 1],
            preds[..., 2],
            preds[..., 3],
            preds[..., 4],
            preds[..., 5:],
        )

        # Build training targets and masks
        (
            tgt_xywh,      # (B, Sy, Sx, A, 4)
            tgt_conf,      # (B, Sy, Sx, A)
            tgt_cls_idx,   # (B, Sy, Sx, A)
            obj_mask,      # bool mask for positive anchors
            noobj_mask,    # bool mask for negative anchors
        ) = build_targets(
            targets=targets,
            anchors=anchors,
            grid_size=Sy,
            img_dim=self.img_dim,
            num_classes=self.num_classes,
            ignore_iou_thr=self.ignore_iou_thr,
            noobj_iou_thr=self.noobj_iou_thr,
            device=device,
        )

        n_pos = obj_mask.sum().float().clamp(min=1.0)
        n_neg = noobj_mask.sum().float().clamp(min=1.0)

        # Coordinate loss for positive anchors only
        if obj_mask.any():
            pred_x = torch.sigmoid(tx[obj_mask])
            pred_y = torch.sigmoid(ty[obj_mask])
            pred_w = tw[obj_mask]
            pred_h = th[obj_mask]
            tgt_x = tgt_xywh[obj_mask][:, 0]
            tgt_y = tgt_xywh[obj_mask][:, 1]
            tgt_w = tgt_xywh[obj_mask][:, 2]
            tgt_h = tgt_xywh[obj_mask][:, 3]

            loss_xy = self.mse(pred_x, tgt_x) + self.mse(pred_y, tgt_y)
            loss_wh = self.mse(pred_w, tgt_w) + self.mse(pred_h, tgt_h)
            loss_coord = self.lambda_coord * (loss_xy + loss_wh)
        else:
            loss_coord = tx.new_zeros(())

        # Objectness and no-objectness losses
        loss_obj = (
            self.bce_obj(to[obj_mask], tgt_conf[obj_mask]
                         ) if obj_mask.any() else to.new_zeros(())
        )
        loss_noobj = (
            self.bce_obj(to[noobj_mask], tgt_conf[noobj_mask]
                         ) if noobj_mask.any() else to.new_zeros(())
        )
        loss_noobj *= self.lambda_noobj

        # Classification loss for positive anchors (multi-label BCE)
        if obj_mask.any():
            cls_target = torch.zeros_like(tcls[obj_mask])
            cls_target.scatter_(1, tgt_cls_idx[obj_mask].unsqueeze(1), 1.0)
            loss_cls = self.bce_cls(tcls[obj_mask], cls_target)
        else:
            loss_cls = tcls.new_zeros(())

        # Normalize losses
        if self.loss_normalize == "batch":
            div = float(B)
        else:  # posneg
            div = n_pos
        loss_coord /= div
        loss_obj /= div
        loss_noobj /= n_neg if self.loss_normalize == "posneg" else float(B)
        loss_cls /= div

        total_loss = loss_coord + loss_obj + loss_noobj + loss_cls
        stat = {
            "coord": loss_coord.detach(),
            "obj": loss_obj.detach(),
            "noobj": loss_noobj.detach(),
            "cls": loss_cls.detach(),
            "pos": n_pos.detach(),
            "neg": n_neg.detach(),
        }

        return total_loss, stat

    def forward(
        self,
        preds: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total multi-scale loss and aggregate statistics.

        Args:
            preds: List of raw prediction tensors from each scale.
            targets: List of ground truth annotations for the batch.

        Returns:
            total_loss: Scalar tensor of total loss aggregated over scales.
            agg: Dictionary of aggregated loss components and counts.
        """
        if len(preds) != len(self.anchor_masks):
            raise ValueError("Length of preds must match anchor_masks")

        total_loss = preds[0].new_zeros(())
        agg: Dict[str, torch.Tensor] = {
            "coord": torch.tensor(0., device=preds[0].device),
            "obj": torch.tensor(0., device=preds[0].device),
            "noobj": torch.tensor(0., device=preds[0].device),
            "cls": torch.tensor(0., device=preds[0].device),
            "pos": torch.tensor(0., device=preds[0].device),
            "neg": torch.tensor(0., device=preds[0].device),
        }

        for pred, mask in zip(preds, self.anchor_masks):
            anchors = self.anchors[mask]
            loss_s, stat_s = self._forward_single_scale(pred, targets, anchors)
            total_loss += loss_s
            for k in agg:
                agg[k] += stat_s[k]

        # Average losses over number of scales for readability
        for k in ("coord", "obj", "noobj", "cls"):
            agg[k] /= len(self.anchor_masks)
        agg["loss"] = total_loss.detach()

        return total_loss, agg
