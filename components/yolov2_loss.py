# yolov2_loss.py
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_targets_yolov2


class YOLOv2Loss(nn.Module):
    """
    Compute YOLOv2 loss: coordinate MSE, objectness BCE, class CrossEntropy.
    """

    def __init__(
        self,
        anchors: List[Tuple[float, float]],  # list of (w, h) in pixels
        num_classes: int,
        img_dim: int = 416,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        ignore_iou_thr: float = 0.7,
    ):
        super().__init__()
        # register raw anchors and normalized dims
        anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anchors_tensor)  # (A, 2)
        self.register_buffer(
            "anchor_w", anchors_tensor[:, 0] / img_dim)  # (A,)
        self.register_buffer(
            "anchor_h", anchors_tensor[:, 1] / img_dim)  # (A,)

        self.num_classes = num_classes
        self.img_dim = img_dim
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.ignore_iou_thr = ignore_iou_thr

        # loss functions
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(reduction="sum")

    def forward(
        self,
        preds: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            preds: Tensor of shape (B, S, S, A*(5+C)), raw output from YOLOv2 head.
            targets: list of dicts with 'boxes' (N x 4 normalized xyxy) and 'labels' (N).
        Returns:
            total_loss and dict of individual loss terms & counts.
        """
        device = preds.device
        B, S, S2, D = preds.shape
        assert S == S2, "Feature map must be square (S x S)"
        A = self.anchors.size(0)
        expected_D = A * (5 + self.num_classes)
        assert D == expected_D, f"Pred channels ({D}) != A*(5+C) ({expected_D})"

        # reshape predictions to (B, S, S, A, 5+C)
        preds = preds.view(B, S, S, A, 5 + self.num_classes)
        tx = preds[..., 0]
        ty = preds[..., 1]
        tw = preds[..., 2]
        th = preds[..., 3]
        tconf = preds[..., 4]
        tcls = preds[..., 5:]  # class logits

        # build targets
        tgt_xywh, tgt_conf, tgt_cls_idx, obj_mask, noobj_mask = build_targets_yolov2(
            targets=targets,
            anchors=self.anchors,
            S=S,
            img_dim=self.img_dim,
            num_classes=self.num_classes,
            ignore_iou_thr=self.ignore_iou_thr,
            device=device,
        )

        # number of positive/negative anchors
        n_pos = obj_mask.sum().clamp(min=1).float()
        n_neg = noobj_mask.sum().clamp(min=1).float()

        # ---------------- coordinate loss ----------------
        # gather only positive predictions
        pos_xywh = tgt_xywh[obj_mask]            # (N_pos, 4)
        px = torch.sigmoid(tx[obj_mask])
        py = torch.sigmoid(ty[obj_mask])
        pw = tw[obj_mask]
        ph = th[obj_mask]

        loss_x = self.mse(px, pos_xywh[:, 0])
        loss_y = self.mse(py, pos_xywh[:, 1])
        loss_w = self.mse(pw, pos_xywh[:, 2])
        loss_h = self.mse(ph, pos_xywh[:, 3])
        loss_coord = self.lambda_coord * \
            (loss_x + loss_y + loss_w + loss_h) / n_pos

        # ---------------- objectness loss ----------------
        # positive objectness
        loss_obj = self.bce(tconf[obj_mask],   tgt_conf[obj_mask]) / n_pos
        # negative objectness (no-object)
        loss_noobj = self.bce(
            tconf[noobj_mask], tgt_conf[noobj_mask]) * self.lambda_noobj / n_neg

        # ---------------- classification loss ----------------
        # (N_pos, C)
        cls_logits_pos = tcls[obj_mask].view(-1, self.num_classes)
        cls_targets = tgt_cls_idx[obj_mask].view(-1)             # (N_pos,)
        loss_cls = self.ce(cls_logits_pos, cls_targets) / n_pos

        # ---------------- total loss ----------------
        total_loss = loss_coord + loss_obj + loss_noobj + loss_cls

        loss_dict = {
            "loss":   total_loss.detach(),
            "coord":  loss_coord.detach(),
            "obj":    loss_obj.detach(),
            "noobj":  loss_noobj.detach(),
            "cls":    loss_cls.detach(),
            "pos":    n_pos.detach(),
            "neg":    n_neg.detach(),
        }

        return total_loss, loss_dict
