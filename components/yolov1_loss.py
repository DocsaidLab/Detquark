import torch
import torch.nn as nn


def bbox_iou(box1, box2, eps: float = 1e-6):
    """
    Compute IoU between two sets of boxes.
    box1, box2: (..., 4) tensors in (x1,y1,x2,y2) format, absolute coords [0,1]
    """
    # Intersection box
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * \
            (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * \
            (box2[..., 3] - box2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter_area + eps
    return inter_area / union


class YOLOv1Loss(nn.Module):
    """
    Classic YOLOv1 loss with 2 bounding boxes per grid cell.
    """

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 80,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5
    ):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.l_coord = lambda_coord
        self.l_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")   # total loss over batch

    @staticmethod
    def _xywh_to_xyxy(xywh):
        """
        Convert (cx,cy,w,h) → (x1,y1,x2,y2); all values ∈ [0,1]
        """
        cx, cy, w, h = xywh.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def forward(self, preds: torch.Tensor, targets):
        """
        Args:
            preds: Tensor[B, S, S, B*5 + C] raw network outputs
            targets: List[dict] (batch length) with keys
                     'boxes': Tensor[N,4] normalized xyxy, 0‑1 range
                     'labels': Tensor[N]

        Returns:
            total_loss, loss_dict
        """
        device = preds.device
        Bsize = preds.size(0)

        # Split preds → boxes/conf/classes
        pred_boxes = preds[..., : self.B * 5].view(
            Bsize, self.S, self.S, self.B, 5
        )                                   # (B,S,S,B,5)
        pred_cls = preds[..., self.B * 5:]  # (B,S,S,C)

        # Prepare ground‑truth tensors
        coord_target = torch.zeros_like(pred_boxes[..., :4])  # (B,S,S,B,4)
        conf_target = torch.zeros_like(pred_boxes[..., 4])    # (B,S,S,B)
        obj_mask = torch.zeros_like(conf_target, dtype=torch.bool)  # (B,S,S,B)
        class_target = torch.zeros_like(pred_cls)             # (B,S,S,C)

        for b, t in enumerate(targets):
            if t["boxes"].numel() == 0:
                continue

            gt_boxes = t["boxes"].to(device)            # (N,4) xyxy
            gt_labels = t["labels"].to(device)          # (N,)

            # Convert GT boxes to (cx,cy,w,h)
            gx1, gy1, gx2, gy2 = gt_boxes.t()
            gcx = (gx1 + gx2) / 2
            gcy = (gy1 + gy2) / 2
            gw = (gx2 - gx1)
            gh = (gy2 - gy1)

            for k in range(gt_boxes.size(0)):
                # Grid indices
                gi = torch.clamp((gcx[k] * self.S).long(), 0, self.S - 1)
                gj = torch.clamp((gcy[k] * self.S).long(), 0, self.S - 1)

                # Predicted boxes at this cell (B,5)
                p_cell = pred_boxes[b, gj, gi]          # (B,5)
                # Convert to xyxy for IoU
                p_xyxy = self._xywh_to_xyxy(
                    torch.stack(
                        (
                            (p_cell[..., 0] + gi.float()) / self.S,
                            (p_cell[..., 1] + gj.float()) / self.S,
                            p_cell[..., 2].pow(2),      # ensure positive
                            p_cell[..., 3].pow(2),
                        ),
                        dim=-1,
                    )
                )
                # Ground‑truth box
                g_xyxy = gt_boxes[k].unsqueeze(0).expand(self.B, 4)
                ious = bbox_iou(p_xyxy, g_xyxy)          # (B,)

                best_box = torch.argmax(ious)            # index 0/1

                obj_mask[b, gj, gi, best_box] = True
                conf_target[b, gj, gi, best_box] = ious[best_box].detach()

                # coordinate target (relative to cell)
                coord_target[b, gj, gi, best_box, 0] = gcx[k] * self.S - gi
                coord_target[b, gj, gi, best_box, 1] = gcy[k] * self.S - gj
                coord_target[b, gj, gi, best_box, 2] = torch.sqrt(gw[k])
                coord_target[b, gj, gi, best_box, 3] = torch.sqrt(gh[k])

                # class target (one‑hot)
                class_target[b, gj, gi, gt_labels[k]] = 1.0

        ### --- Losses --- ###
        # Coordinate loss
        coord_pred = pred_boxes[..., :4]
        coord_loss = self.mse(
            coord_pred[obj_mask], coord_target[obj_mask]
        ) * self.l_coord

        # Objectness loss
        conf_pred = pred_boxes[..., 4]
        obj_loss = self.mse(
            conf_pred[obj_mask], conf_target[obj_mask]
        )

        # No‑object loss
        noobj_loss = self.mse(
            conf_pred[~obj_mask], conf_target[~obj_mask]
        ) * self.l_noobj

        # Classification loss
        cls_loss = self.mse(
            pred_cls[obj_mask.any(dim=-1)],  # (B,S,S)
            class_target[obj_mask.any(dim=-1)],
        )

        total = coord_loss + obj_loss + noobj_loss + cls_loss
        loss_dict = dict(
            loss=total.detach(),
            coord=coord_loss.detach(),
            obj=obj_loss.detach(),
            noobj=noobj_loss.detach(),
            cls=cls_loss.detach(),
        )
        return total, loss_dict
