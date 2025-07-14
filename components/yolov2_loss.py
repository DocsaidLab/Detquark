from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn

from .utils import bbox_iou


class YOLOv2Loss(nn.Module):
    """YOLOv2 loss module combining coordinate, objectness, and classification losses.

    Attributes:
        anchors (Tensor): Anchor box sizes (A, 2) in pixels.
        anchor_w (Tensor): Anchor widths normalized by image dimension.
        anchor_h (Tensor): Anchor heights normalized by image dimension.
        num_classes (int): Number of object classes.
        img_dim (int): Input image dimension in pixels.
        lambda_coord (float): Weight for coordinate loss.
        lambda_noobj (float): Weight for no-objectness loss.
        ignore_iou_thr (float): IoU threshold to ignore objectness loss.
        noobj_iou_thr (float): IoU threshold to suppress no-objectness loss.
        mse (nn.MSELoss): Mean squared error loss with sum reduction.
        bce (nn.BCEWithLogitsLoss): Binary cross entropy loss with sum reduction.
        ce (nn.CrossEntropyLoss): Cross entropy loss with sum reduction.
        loss_normalize (Literal["batch","posneg"]): Loss normalization mode.
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
        """Initializes YOLOv2 loss parameters and buffers.

        Args:
            anchors: List of anchor sizes in pixels [(w, h), ...].
            num_classes: Number of object classes.
            img_dim: Input image size in pixels.
            lambda_coord: Weight factor for coordinate regression loss.
            lambda_noobj: Weight factor for no-objectness loss.
            ignore_iou_thr: IoU threshold above which no-object loss is ignored.
            noobj_iou_thr: IoU threshold below which no-object loss is active.
            loss_normalize: Normalization strategy for losses ("batch" or "posneg").
        """
        super().__init__()

        anc = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anc)
        self.register_buffer("anchor_w", anc[:, 0] / img_dim)
        self.register_buffer("anchor_h", anc[:, 1] / img_dim)

        self.num_classes = num_classes
        self.img_dim = img_dim
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.ignore_iou_thr = ignore_iou_thr
        self.noobj_iou_thr = noobj_iou_thr

        if not (0.0 <= self.noobj_iou_thr <= self.ignore_iou_thr < 1.0):
            raise ValueError(
                "noobj_iou_thr must satisfy 0.0 <= noobj_iou_thr <= ignore_iou_thr < 1.0"
            )

        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(reduction="sum")

        if loss_normalize not in ("batch", "posneg"):
            raise ValueError("loss_normalize must be 'batch' or 'posneg'")
        self.loss_normalize = loss_normalize

    def forward(
        self,
        preds: torch.Tensor,  # shape: (B, S, S, A*(5 + C))
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and its components from predictions and targets.

        Args:
            preds: Raw network predictions tensor of shape (B, S, S, A*(5 + C)).
            targets: List of ground truth annotations per batch item.

        Returns:
            total_loss: Scalar total loss tensor.
            loss_dict: Dict containing individual loss components and counts.
        """
        device = preds.device
        B, S, S2, D = preds.shape

        if S != S2:
            raise ValueError("Feature map must be square")

        A = self.anchors.size(0)
        expected_D = A * (5 + self.num_classes)
        if D != expected_D:
            raise ValueError(f"Expected last dim {expected_D}, got {D}")

        preds = preds.view(B, S, S, A, 5 + self.num_classes)
        tx = preds[..., 0]
        ty = preds[..., 1]
        tw = preds[..., 2]
        th = preds[..., 3]
        to = preds[..., 4]
        tcls = preds[..., 5:]

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

        n_pos = obj_mask.sum().float().clamp(min=1.0)
        n_neg = noobj_mask.sum().float().clamp(min=1.0)

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

        loss_obj = self.bce(to[obj_mask], tgt_conf[obj_mask]) \
            if obj_mask.any() else to.new_zeros(())
        loss_noobj = self.bce(to[noobj_mask], tgt_conf[noobj_mask]) \
            if noobj_mask.any() else to.new_zeros(())
        loss_noobj *= self.lambda_noobj

        if obj_mask.any():
            cls_logits = tcls[obj_mask].reshape(-1, self.num_classes)
            cls_targets = tgt_cls_idx[obj_mask].reshape(-1)
            loss_cls = self.ce(cls_logits, cls_targets)
        else:
            loss_cls = tcls.new_zeros(())

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
        torch.Tensor,
    ]:
        """Construct target tensors and masks for training.

        Args:
            targets: List of ground truth annotations per batch item.
            anchors: Anchor box sizes (A, 2).
            S: Grid size.
            img_dim: Image dimension in pixels.
            num_classes: Number of object classes.
            ignore_iou_thr: IoU threshold to ignore no-object loss.
            noobj_iou_thr: IoU threshold to suppress no-object loss.
            device: Device for target tensors.

        Returns:
            target_xywh: Tensor of shape (B, S, S, A, 4) with target box offsets.
            target_conf: Objectness target tensor (B, S, S, A).
            target_cls_idx: Class index targets (B, S, S, A).
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

        target_xywh = torch.zeros((B, S, S, A, 4), device=device)
        target_conf = torch.zeros((B, S, S, A), device=device)
        target_cls_idx = torch.zeros(
            (B, S, S, A), dtype=torch.long, device=device)
        obj_mask = torch.zeros((B, S, S, A), dtype=torch.bool, device=device)
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

            gi = (cx * S).floor().clamp(0, S - 1).long()
            gj = (cy * S).floor().clamp(0, S - 1).long()

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

                target_xywh[b_idx, j, i, best_a, 0] = cx[k] * S - i
                target_xywh[b_idx, j, i, best_a, 1] = cy[k] * S - j
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
