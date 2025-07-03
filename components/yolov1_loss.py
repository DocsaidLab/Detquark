from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .utils import bbox_iou


class YOLOv1Loss(nn.Module):
    """
    YOLOv1 loss: squared error for bounding box coordinates, objectness, and class probabilities.
    """

    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 80,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5
    ):
        super().__init__()
        self.S = grid_size   # number of grid cells per side
        self.B = num_boxes   # number of boxes per cell
        self.C = num_classes  # number of classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        # MSE loss for all components
        self.mse_loss = nn.MSELoss(reduction="sum")

    @staticmethod
    def _xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
        """
        Convert box format from (cx, cy, w, h) to (x1, y1, x2, y2).
        All values expected in normalized [0,1] range.
        """
        cx, cy, w, h = box_xywh.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(
        self,
        preds: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            preds: Tensor of shape (batch, S, S, B*5 + C)
            targets: list of dicts with keys:
                     'boxes': (N,4) xyxy normalized,
                     'labels': (N,) long
        Returns:
            total_loss: scalar tensor
            loss_components: dict of individual loss terms
        """
        batch_size, S1, S2, D = preds.shape

        # Verify dimensions
        if S1 != self.S or S2 != self.S:
            raise ValueError(
                f"Grid size mismatch: expected {self.S}, got {S1}x{S2}")

        expected_D = self.B * 5 + self.C
        if D != expected_D:
            raise ValueError(
                f"Pred dimension mismatch: expected {expected_D}, got {D}")

        device = preds.device

        # Split class and box predictions
        # (batch, S, S, C)
        pred_cls = preds[..., self.B * 5:]
        pred_boxes = preds[..., : self.B * 5].view(
            batch_size, self.S, self.S, self.B, 5
        )  # (batch, S, S, B, 5)

        # Extract individual components
        pred_tx = pred_boxes[..., 0]
        pred_ty = pred_boxes[..., 1]
        pred_tw = pred_boxes[..., 2]
        pred_th = pred_boxes[..., 3]
        pred_to = pred_boxes[..., 4]

        # Initialize target tensors
        coord_target = torch.zeros_like(
            pred_boxes[..., :4], device=device)  # (batch, S, S, B, 4)
        conf_target = torch.zeros_like(
            pred_boxes[..., 4], device=device)     # (batch, S, S, B)
        obj_mask = torch.zeros_like(
            conf_target, dtype=torch.bool)           # (batch, S, S, B)
        class_target = torch.zeros_like(
            pred_cls, device=device)             # (batch, S, S, C)

        # Build targets per image
        for b_idx, sample in enumerate(targets):
            gt_boxes = sample.get("boxes", torch.empty((0, 4), device=device))
            gt_labels = sample.get("labels", torch.empty(
                (0,), dtype=torch.long, device=device))
            if gt_boxes.numel() == 0:
                continue

            # Convert GT boxes from xyxy to cx, cy, w, h
            x1, y1, x2, y2 = gt_boxes.unbind(-1)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = (x2 - x1).clamp(min=1e-6)
            h = (y2 - y1).clamp(min=1e-6)

            # Compute grid cell indices
            grid_x = (cx * self.S).floor().clamp(0, self.S - 1).long()
            grid_y = (cy * self.S).floor().clamp(0, self.S - 1).long()

            # Iterate ground-truth boxes
            for i in range(gt_boxes.size(0)):
                gi = grid_x[i]
                gj = grid_y[i]

                # Predicted boxes in this cell: shape (B, 5)
                cell_preds = pred_boxes[b_idx, gj, gi]  # (B,5)

                # Reconstruct predicted box corners for IoU
                # x_center = sigmoid(tx) + cell_index  all / S
                px = (torch.sigmoid(cell_preds[:, 0]) + gi.float()) / self.S
                py = (torch.sigmoid(cell_preds[:, 1]) + gj.float()) / self.S
                pw = cell_preds[:, 2].pow(2)
                ph = cell_preds[:, 3].pow(2)
                pred_box_xywh = torch.stack([px, py, pw, ph], dim=-1)
                pred_box_xyxy = self._xywh_to_xyxy(pred_box_xywh)

                # Prepare GT box for IoU
                gt_box_exp = gt_boxes[i].unsqueeze(0).expand(self.B, 4)
                ious = bbox_iou(pred_box_xyxy, gt_box_exp)  # (B,)
                best = torch.argmax(ious)

                # Mark responsible box
                obj_mask[b_idx, gj, gi, best] = True
                # Confidence target = IoU of responsible box
                conf_target[b_idx, gj, gi, best] = ious[best].detach()

                # Coordinate targets relative to cell
                coord_target[b_idx, gj, gi, best, 0] = cx[i] * self.S - gi
                coord_target[b_idx, gj, gi, best, 1] = cy[i] * self.S - gj
                coord_target[b_idx, gj, gi, best, 2] = torch.sqrt(w[i])
                coord_target[b_idx, gj, gi, best, 3] = torch.sqrt(h[i])

                # One-hot class target for this cell
                class_target[b_idx, gj, gi, gt_labels[i]] = 1.0

        # Compute masks for positive and negative samples
        # (batch, S, S, B)
        pos_mask = obj_mask
        neg_mask = ~obj_mask
        # (batch, S, S)
        cell_mask = obj_mask.any(dim=-1)

        # Number of positive/negative anchors (avoid zero)
        N_pos = pos_mask.sum().clamp(min=1).float()
        N_neg = neg_mask.sum().clamp(min=1).float()
        N_cell = cell_mask.sum().clamp(min=1).float()

        # --- Coordinate loss (for x, y, w, h) ---
        loss_x = self.mse_loss(torch.sigmoid(
            pred_tx[pos_mask]), coord_target[..., 0][pos_mask])
        loss_y = self.mse_loss(torch.sigmoid(
            pred_ty[pos_mask]), coord_target[..., 1][pos_mask])
        loss_w = self.mse_loss(
            pred_tw[pos_mask], coord_target[..., 2][pos_mask])
        loss_h = self.mse_loss(
            pred_th[pos_mask], coord_target[..., 3][pos_mask])
        loss_coord = self.lambda_coord * \
            (loss_x + loss_y + loss_w + loss_h) / N_pos

        # --- Objectness loss ---
        loss_obj = self.mse_loss(
            torch.sigmoid(pred_to[pos_mask]), conf_target[pos_mask]) / N_pos
        loss_noobj = self.lambda_noobj * self.mse_loss(
            torch.sigmoid(pred_to[neg_mask]), conf_target[neg_mask]) / N_neg

        # --- Classification loss (for cells with objects) ---
        # pred_cls is (batch, S, S, C)
        cls_pred = torch.softmax(pred_cls[cell_mask], dim=-1)  # (N_cell, C)
        cls_tgt = class_target[cell_mask]                      # (N_cell, C)
        loss_cls = self.mse_loss(cls_pred, cls_tgt) / N_cell

        # Total loss
        total_loss = loss_coord + loss_obj + loss_noobj + loss_cls

        # Return loss and components
        loss_dict = {
            "loss": total_loss.detach(),
            "coord": loss_coord.detach(),
            "obj": loss_obj.detach(),
            "noobj": loss_noobj.detach(),
            "cls": loss_cls.detach(),
        }

        return total_loss, loss_dict
