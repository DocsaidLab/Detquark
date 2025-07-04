from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_targets_yolov2


class YOLOv2Loss(nn.Module):
    """
    YOLOv2 loss combining coordinate regression, objectness, and classification.

    Attributes:
        anchors (Tensor[A, 2]): Anchor box sizes in pixels.
        anchor_w (Tensor[A]): Anchor widths normalized by image dimension.
        anchor_h (Tensor[A]): Anchor heights normalized by image dimension.
        num_classes (int): Number of object classes.
        img_dim (int): Input image size in pixels.
        lambda_coord (float): Weight for coordinate loss.
        lambda_noobj (float): Weight for no-object confidence loss.
        ignore_iou_thr (float): IoU threshold to ignore negative anchors.
        mse (MSELoss): Sum-reduction mean squared error.
        bce (BCEWithLogitsLoss): Sum-reduction binary cross-entropy.
        ce (CrossEntropyLoss): Sum-reduction cross-entropy for classification.
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
        """
        Initialize YOLOv2Loss.

        Args:
            anchors (List[Tuple[float, float]]): Anchor sizes (w,h) in pixels.
            num_classes (int): Number of object classes.
            img_dim (int): Input image size (pixels). Default 416.
            lambda_coord (float): Weight for coordinate loss. Default 5.0.
            lambda_noobj (float): Weight for no-object loss. Default 0.5.
            ignore_iou_thr (float): IoU threshold to ignore negatives. Default 0.7.
        """
        super().__init__()
        anc = torch.tensor(anchors, dtype=torch.float32)
        # Raw anchor sizes for reference
        self.register_buffer("anchors", anc)         # (A, 2)
        # Normalized anchor dims in [0,1]
        self.register_buffer("anchor_w", anc[:, 0] / img_dim)  # (A,)
        self.register_buffer("anchor_h", anc[:, 1] / img_dim)  # (A,)

        self.num_classes = num_classes
        self.img_dim = img_dim
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.ignore_iou_thr = ignore_iou_thr

        # Loss functions with sum reduction for stable scaling
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(reduction="sum")

    def forward(
        self,
        preds: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute YOLOv2 loss.

        Args:
            preds (Tensor[B, S, S, A*(5+C)]): Raw output from prediction head.
            targets (List[dict]): Ground-truth per image, each with:
                - 'boxes' (Tensor[N,4]): normalized xyxy boxes.
                - 'labels' (Tensor[N]): class indices.

        Returns:
            total_loss (Tensor): Scalar sum of all loss terms.
            loss_dict (dict): Individual losses and anchor counts:
                'coord', 'obj', 'noobj', 'cls', 'pos', 'neg', plus 'loss'.

        Raises:
            AssertionError: If feature map is not square or channel size mismatches.
        """
        device = preds.device
        B, S, S2, D = preds.shape
        assert S == S2, "Feature map must be square (S x S)"
        A = self.anchors.size(0)
        expected_D = A * (5 + self.num_classes)
        assert D == expected_D, f"Pred channels ({D}) != A*(5+C) ({expected_D})"

        # ------------------------------------------------------------------
        # 1. Reshape predictions to separate anchors
        # ------------------------------------------------------------------
        preds = preds.view(B, S, S, A, 5 + self.num_classes)
        tx, ty, tw, th, to = (
            preds[..., 0],
            preds[..., 1],
            preds[..., 2],
            preds[..., 3],
            preds[..., 4],
        )
        tcls = preds[..., 5:]  # (B, S, S, A, C)

        # ------------------------------------------------------------------
        # 2. Build training targets and masks
        # ------------------------------------------------------------------
        tgt_xywh, tgt_conf, tgt_cls_idx, obj_mask, noobj_mask = build_targets_yolov2(
            targets=targets,
            anchors=self.anchors,
            S=S,
            img_dim=self.img_dim,
            num_classes=self.num_classes,
            ignore_iou_thr=self.ignore_iou_thr,
            device=device,
        )
        # Ensure at least one positive/negative to avoid divide-by-zero
        n_pos = obj_mask.sum().clamp(min=1).float()
        n_neg = noobj_mask.sum().clamp(min=1).float()

        # ------------------------------------------------------------------
        # 3. Coordinate loss (only positive anchors)
        # ------------------------------------------------------------------
        pos_xywh = tgt_xywh[obj_mask]      # (N_pos, 4)
        px = torch.sigmoid(tx[obj_mask])   # predict cx
        py = torch.sigmoid(ty[obj_mask])   # predict cy
        pw = tw[obj_mask]                  # predict sqrt(w)
        ph = th[obj_mask]                  # predict sqrt(h)

        loss_x = self.mse(px, pos_xywh[:, 0])
        loss_y = self.mse(py, pos_xywh[:, 1])
        loss_w = self.mse(pw, pos_xywh[:, 2])
        loss_h = self.mse(ph, pos_xywh[:, 3])
        loss_coord = self.lambda_coord * \
            (loss_x + loss_y + loss_w + loss_h) / n_pos

        # ------------------------------------------------------------------
        # 4. Objectness loss
        # ------------------------------------------------------------------
        loss_obj = self.bce(to[obj_mask],   tgt_conf[obj_mask]) / n_pos
        loss_noobj = self.bce(
            to[noobj_mask], tgt_conf[noobj_mask]) * self.lambda_noobj / n_neg

        # ------------------------------------------------------------------
        # 5. Classification loss (only positive anchors)
        # ------------------------------------------------------------------
        cls_logits_pos = tcls[obj_mask].view(-1, self.num_classes)
        cls_targets = tgt_cls_idx[obj_mask].view(-1)
        loss_cls = self.ce(cls_logits_pos, cls_targets) / n_pos

        # ------------------------------------------------------------------
        # 6. Total loss and reporting
        # ------------------------------------------------------------------
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
