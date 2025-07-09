from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn

from .utils import bbox_iou


class YOLOv2Loss(nn.Module):
    """
    YOLOv2 loss function combining localization regression, objectness confidence, and classification.

    Args:
        anchors (List[Tuple[float, float]]): Anchor box dimensions (width, height) in pixels.
        num_classes (int): Total number of object classes.
        img_dim (int, optional): Input image size (height and width) in pixels. Default: 416.
        lambda_coord (float, optional): Weighting factor for the coordinate regression loss. Default: 5.0.
        lambda_noobj (float, optional): Weighting factor for the no-object confidence loss. Default: 0.5.
        ignore_iou_thr (float, optional): IoU threshold above which anchors are treated as positives and excluded from no-object loss. Default: 0.7.
        noobj_iou_thr (float, optional): IoU threshold below which anchors are treated as negatives and included in no-object loss; anchors with IoU between the two thresholds are ignored. Default: 0.1.
        loss_normalize (Literal["batch", "posneg"], optional):
            - "batch": normalize by batch size (stable, default).
            - "posneg": normalize coordinate and objectness losses by number of positive/negative anchors.
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
        anc = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anc)               # (A,2) in pixels
        self.register_buffer("anchor_w", anc[:, 0] / img_dim)
        self.register_buffer("anchor_h", anc[:, 1] / img_dim)

        self.num_classes = num_classes
        self.img_dim = img_dim
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.ignore_iou_thr = ignore_iou_thr
        self.noobj_iou_thr = noobj_iou_thr
        assert 0.0 <= self.noobj_iou_thr <= self.ignore_iou_thr < 1.0

        # ── loss function handles ──────────────────────────────────────────
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(reduction="sum")

        # normalisation mode
        assert loss_normalize in ("batch", "posneg")
        self.loss_normalize = loss_normalize

    # --------------------------------------------------------------------- #
    #                               forward                                 #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        preds: torch.Tensor,                       # (B,S,S,A*(5+C))
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = preds.device
        B, S, S2, D = preds.shape
        assert S == S2, "Feature map must be square"
        A = self.anchors.size(0)
        expected_D = A * (5 + self.num_classes)
        assert D == expected_D, f"pred channels={D}, expected={expected_D}"

        # reshape to (B,S,S,A,5+C)
        preds = preds.view(B, S, S, A, 5 + self.num_classes)
        tx = preds[..., 0]
        ty = preds[..., 1]
        tw = preds[..., 2]
        th = preds[..., 3]
        to = preds[..., 4]
        tcls = preds[..., 5:]

        # build targets / masks
        tgt_xywh, tgt_conf, tgt_cls_idx, obj_mask, noobj_mask = \
            self.build_targets_yolov2(
                targets=targets,
                anchors=self.anchors,
                S=S,
                img_dim=self.img_dim,
                num_classes=self.num_classes,
                ignore_iou_thr=self.ignore_iou_thr,
                noobj_iou_thr=self.noobj_iou_thr,
                device=device,
            )

        n_pos = obj_mask.sum().float().clamp(min=1.0)   # avoid /0
        n_neg = noobj_mask.sum().float().clamp(min=1.0)

        # --------------------------- coordinate ---------------------------
        if obj_mask.any():
            pos_xywh = tgt_xywh[obj_mask]                 # (N_pos,4)
            pred_x = torch.sigmoid(tx[obj_mask])          # ∈(0,1)
            pred_y = torch.sigmoid(ty[obj_mask])          # ∈(0,1)
            pred_w = tw[obj_mask]
            pred_h = th[obj_mask]

            loss_coord = self.lambda_coord * (
                self.mse(pred_x, pos_xywh[:, 0]) +
                self.mse(pred_y, pos_xywh[:, 1]) +
                self.mse(pred_w, pos_xywh[:, 2]) +
                self.mse(pred_h, pos_xywh[:, 3])
            )
        else:
            loss_coord = torch.tensor(0., device=device)

        # ---------------------------- objectness --------------------------
        loss_obj = self.bce(to[obj_mask], tgt_conf[obj_mask]) if obj_mask.any(
        ) else torch.tensor(0., device=device)
        loss_noobj = self.bce(to[noobj_mask], tgt_conf[noobj_mask]) \
            if noobj_mask.any() else torch.tensor(0., device=device)
        loss_noobj *= self.lambda_noobj

        # ----------------------------- class ------------------------------
        if obj_mask.any():
            cls_logits = tcls[obj_mask].view(-1, self.num_classes)
            cls_tgt = tgt_cls_idx[obj_mask].view(-1)
            loss_cls = self.ce(cls_logits, cls_tgt)
        else:
            loss_cls = torch.tensor(0., device=device)

        # ----------------- normalisation: batch or pos/neg ----------------
        if self.loss_normalize == "batch":
            div = float(B)
            loss_coord /= div
            loss_obj /= div
            loss_noobj /= div
            loss_cls /= div
        else:  # "posneg"
            loss_coord /= n_pos
            loss_obj /= n_pos
            loss_noobj /= n_neg
            loss_cls /= n_pos

        total_loss = loss_coord + loss_obj + loss_noobj + loss_cls

        loss_dict = {
            "loss":  total_loss.detach(),
            "coord": loss_coord.detach(),
            "obj":   loss_obj.detach(),
            "noobj": loss_noobj.detach(),
            "cls":   loss_cls.detach(),
            "pos":   n_pos.detach(),
            "neg":   n_neg.detach(),
        }
        return total_loss, loss_dict

    # --------------------------------------------------------------------- #
    #                         target generation                             #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def build_targets_yolov2(
        self,
        targets: List[Dict[str, torch.Tensor]],
        anchors: torch.Tensor,            # (A,2) in pixels
        S: int,
        img_dim: int,
        num_classes: int,
        ignore_iou_thr: float,
        noobj_iou_thr: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Return:
            target_xywh  (B,S,S,A,4): (tx,ty,tw,th) offsets where tx,ty∈[0,1]
            target_conf  (B,S,S,A)
            target_cls_idx (B,S,S,A) : non‑obj 置 0；實際是否計算看 obj_mask
            obj_mask, noobj_mask (bool)
        """
        B = len(targets)
        A = anchors.size(0)

        # anchor boxes (normalized, centered at (0,0))
        anchor_w = anchors[:, 0] / img_dim
        anchor_h = anchors[:, 1] / img_dim
        anchor_boxes = torch.stack(
            (-anchor_w / 2, -anchor_h / 2, anchor_w / 2, anchor_h / 2),
            dim=1
        ).to(device)                                  # (A,4)

        # allocate
        target_xywh = torch.zeros((B, S, S, A, 4), device=device)
        target_conf = torch.zeros((B, S, S, A), device=device)
        # 以 0 填充（合法 class id），真正是否用到由 obj_mask 決定
        target_cls_idx = torch.zeros(
            (B, S, S, A), dtype=torch.long, device=device)

        obj_mask = torch.zeros((B, S, S, A), dtype=torch.bool, device=device)
        noobj_mask = torch.ones_like(obj_mask)

        # ── encode each image ──────────────────────────────────────────────
        for b_idx, sample in enumerate(targets):
            boxes = sample.get("boxes", torch.empty(0, 4)).to(device)
            labels = sample.get("labels", torch.empty(
                0, dtype=torch.long)).to(device)

            if boxes.numel() == 0:
                continue
            assert labels.max() < num_classes, "label id ≥ num_classes"

            # (cx,cy,w,h), normalized
            x1, y1, x2, y2 = boxes.unbind(1)
            cx, cy = (x1+x2)*0.5, (y1+y2)*0.5
            w, h = (x2-x1).clamp(min=1e-6), (y2-y1).clamp(min=1e-6)

            # grid indices
            gi = (cx * S).floor().clamp(0, S-1).long()
            gj = (cy * S).floor().clamp(0, S-1).long()

            for k in range(boxes.size(0)):
                i, j = gi[k], gj[k]

                # IoU with each anchor (shape only)
                gt_box = torch.tensor(
                    [-w[k]/2, -h[k]/2,  w[k]/2,  h[k]/2],
                    dtype=boxes.dtype, device=device
                ).unsqueeze(0)
                ious = bbox_iou(anchor_boxes, gt_box).view(-1)
                best_a = torch.argmax(ious)

                # 若該 anchor 已標為正樣本 -> 跳過 (YOLOv2 簡化策略)
                if obj_mask[b_idx, j, i, best_a]:
                    continue

                # 正樣本
                obj_mask[b_idx, j, i, best_a] = True
                noobj_mask[b_idx, j, i, best_a] = False

                # --- offsets ---
                target_xywh[b_idx, j, i, best_a,
                            0] = cx[k]*S - i   # tx ∈ [0,1]
                target_xywh[b_idx, j, i, best_a,
                            1] = cy[k]*S - j   # ty ∈ [0,1]
                target_xywh[b_idx, j, i, best_a, 2] = torch.log(
                    w[k] / anchor_w[best_a] + 1e-16)
                target_xywh[b_idx, j, i, best_a, 3] = torch.log(
                    h[k] / anchor_h[best_a] + 1e-16)

                target_conf[b_idx, j, i, best_a] = 1.0
                target_cls_idx[b_idx, j, i, best_a] = labels[k]

                # ---- no‑obj ignore zone ----------------------------------
                # IoU > ignore_iou_thr → 不算 no‑obj
                mask_high = ious > ignore_iou_thr
                # IoU > noobj_iou_thr → 也不計 no‑obj（排除「灰區」）
                mask_mid = ious > noobj_iou_thr
                suppress_mask = mask_high | mask_mid
                if suppress_mask.any():
                    noobj_mask[b_idx, j, i, suppress_mask] = False

        return target_xywh, target_conf, target_cls_idx, obj_mask, noobj_mask
