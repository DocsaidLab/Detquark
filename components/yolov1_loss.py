from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .utils import bbox_iou


class YOLOv1Loss(nn.Module):
    """
    Compute YOLOv1 loss, combining localization, objectness, and classification errors.

    Attributes:
        S (int): Number of grid cells per side.
        B (int): Number of bounding boxes predicted per cell.
        C (int): Number of object classes.
        lambda_coord (float): Weight for coordinate (xywh) loss.
        lambda_noobj (float): Weight for no-object confidence loss.
        mse_loss (nn.MSELoss): Mean squared error loss (sum reduction).
    """

    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 80,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5
    ):
        """
        Initialize YOLOv1Loss.

        Args:
            grid_size (int): Number of grid cells per side (S). Default 7.
            num_boxes (int): Number of boxes per cell (B). Default 2.
            num_classes (int): Number of object classes (C). Default 80.
            lambda_coord (float): Weight for coordinate loss. Default 5.0.
            lambda_noobj (float): Weight for no-object confidence loss. Default 0.5.
        """
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss(reduction="sum")

    @staticmethod
    def _xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

        Args:
            box_xywh (Tensor[..., 4]): Boxes in (cx, cy, w, h) normalized format.

        Returns:
            Tensor[..., 4]: Boxes in (x1, y1, x2, y2) normalized format.
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
        Compute total YOLOv1 loss and its components.

        Args:
            preds (Tensor): Predictions of shape (batch, S, S, B*5 + C).
            targets (List[dict]): Ground-truth list, each dict contains:
                - 'boxes' (Tensor[N, 4]): normalized xyxy boxes.
                - 'labels' (Tensor[N]): class indices.

        Returns:
            total_loss (Tensor): Scalar tensor of summed loss.
            loss_components (dict): Individual loss terms:
                'coord', 'obj', 'noobj', 'cls', plus 'loss' (total).

        Raises:
            ValueError: If prediction grid size or depth mismatch configuration.
        """
        batch_size, S1, S2, D = preds.shape

        # Validate grid and prediction dimensions
        if (S1, S2) != (self.S, self.S):
            raise ValueError(
                f"Grid size mismatch: expected {self.S} x {self.S}, got {S1} x {S2}")
        expected_D = self.B * 5 + self.C
        if D != expected_D:
            raise ValueError(
                f"Pred depth mismatch: expected {expected_D}, got {D}")

        device = preds.device

        # ------------------------------------------------------------------
        # 1. Split predictions into class scores and box parameters
        # ------------------------------------------------------------------
        # (batch, S, S, C)
        pred_cls = preds[..., self.B * 5:]
        pred_boxes = preds[..., : self.B * 5].view(
            batch_size, self.S, self.S, self.B, 5
            # (batch, S, S, B, 5)
        )
        tx, ty, tw, th, to = (
            pred_boxes[..., 0],
            pred_boxes[..., 1],
            pred_boxes[..., 2],
            pred_boxes[..., 3],
            pred_boxes[..., 4],
        )

        # ------------------------------------------------------------------
        # 2. Prepare target tensors and masks
        # ------------------------------------------------------------------
        coord_target = torch.zeros_like(
            pred_boxes[..., :4], device=device)   # (batch, S, S, B, 4)
        conf_target = torch.zeros_like(
            to, device=device)                    # (batch, S, S, B)
        obj_mask = torch.zeros_like(
            conf_target, dtype=torch.bool)        # positive mask
        class_target = torch.zeros_like(
            pred_cls, device=device)              # (batch, S, S, C)

        # ------------------------------------------------------------------
        # 3. Assign ground-truth boxes to responsible anchors
        # ------------------------------------------------------------------
        for b_idx, sample in enumerate(targets):
            gt_boxes = sample.get("boxes", torch.empty((0, 4), device=device))
            gt_labels = sample.get("labels", torch.empty(
                (0,), dtype=torch.long, device=device))
            if gt_boxes.numel() == 0:
                continue

            # Convert GT boxes to center format
            x1, y1, x2, y2 = gt_boxes.unbind(-1)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = (x2 - x1).clamp(min=1e-6)
            h = (y2 - y1).clamp(min=1e-6)

            # Determine grid cell indices
            grid_x = (cx * self.S).floor().clamp(0, self.S - 1).long()
            grid_y = (cy * self.S).floor().clamp(0, self.S - 1).long()

            for i in range(gt_boxes.size(0)):
                gi, gj = grid_x[i], grid_y[i]
                cell_preds = pred_boxes[b_idx, gj, gi]  # (B, 5)

                # Decode predicted boxes for IoU comparison
                px = (torch.sigmoid(cell_preds[:, 0]) + gi.float()) / self.S
                py = (torch.sigmoid(cell_preds[:, 1]) + gj.float()) / self.S
                pw = cell_preds[:, 2].pow(2)
                ph = cell_preds[:, 3].pow(2)
                pred_xywh = torch.stack([px, py, pw, ph], dim=-1)
                pred_xyxy = self._xywh_to_xyxy(pred_xywh)

                # Find best anchor by IoU
                gt_exp = gt_boxes[i].unsqueeze(0).expand(self.B, 4)
                ious = bbox_iou(pred_xyxy, gt_exp)
                best = torch.argmax(ious)

                # Mark positive anchor and set confidence target
                obj_mask[b_idx, gj, gi, best] = True
                conf_target[b_idx, gj, gi, best] = ious[best].detach()

                # Coordinate regression targets
                coord_target[b_idx, gj, gi, best, 0] = cx[i] * self.S - gi
                coord_target[b_idx, gj, gi, best, 1] = cy[i] * self.S - gj
                coord_target[b_idx, gj, gi, best, 2] = torch.sqrt(w[i])
                coord_target[b_idx, gj, gi, best, 3] = torch.sqrt(h[i])

                # One-hot class target
                class_target[b_idx, gj, gi, gt_labels[i]] = 1.0

        # ------------------------------------------------------------------
        # 4. Compute loss components
        # ------------------------------------------------------------------
        pos_mask = obj_mask
        neg_mask = ~obj_mask
        cell_mask = obj_mask.any(dim=-1)

        N_pos = pos_mask.sum().clamp(min=1).float()
        N_neg = neg_mask.sum().clamp(min=1).float()
        N_cell = cell_mask.sum().clamp(min=1).float()

        # Localization loss (x, y, w, h)
        loss_xy = self.mse_loss(torch.sigmoid(tx[pos_mask]), coord_target[..., 0][pos_mask]) \
            + self.mse_loss(torch.sigmoid(ty[pos_mask]),
                            coord_target[..., 1][pos_mask])
        loss_wh = self.mse_loss(tw[pos_mask], coord_target[..., 2][pos_mask]) \
            + self.mse_loss(th[pos_mask], coord_target[..., 3][pos_mask])
        loss_coord = self.lambda_coord * (loss_xy + loss_wh) / N_pos

        # Objectness loss
        loss_obj = self.mse_loss(
            torch.sigmoid(to[pos_mask]), conf_target[pos_mask]) / N_pos
        loss_noobj = self.lambda_noobj * self.mse_loss(
            torch.sigmoid(to[neg_mask]), conf_target[neg_mask]) / N_neg

        # Classification loss for cells with objects
        cls_pred = torch.softmax(pred_cls[cell_mask], dim=-1)
        cls_tgt = class_target[cell_mask]
        loss_cls = self.mse_loss(cls_pred, cls_tgt) / N_cell

        total_loss = loss_coord + loss_obj + loss_noobj + loss_cls
        loss_dict = {
            "loss": total_loss.detach(),
            "coord": loss_coord.detach(),
            "obj": loss_obj.detach(),
            "noobj": loss_noobj.detach(),
            "cls": loss_cls.detach(),
        }

        return total_loss, loss_dict
