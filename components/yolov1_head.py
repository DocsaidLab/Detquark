from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


class YOLOv1Head(nn.Module):
    """
    YOLOv1 detection head for object detection.
    Predicts B bounding boxes and C class probabilities per SxS grid cell.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_bboxes: int = 2,
        grid_size: int = 7,
        **kwargs
    ):
        super().__init__()
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        # feature extractor: conv → batchnorm → LeakyReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        # fully‑connected head: flatten → FC → LeakyReLU + Dropout → final FC
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),
        )

        # pre‑compute grid offsets for decoding
        gy, gx = torch.meshgrid(
            torch.arange(self.S), torch.arange(self.S), indexing="ij"
        )
        # grid_x, grid_y have shape (1, S, S, 1)
        self.register_buffer("grid_x", gx.unsqueeze(0).unsqueeze(-1).float())
        self.register_buffer("grid_y", gy.unsqueeze(0).unsqueeze(-1).float())
        self.cell_size = 1.0 / self.S

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of intermediate features, take the last one (B, C, S, S)
        Returns:
            raw predictions of shape (B, S, S, B*5 + C)
        """
        x = features[-1]
        x = self.conv(x)
        x = self.fc(x)
        # reshape to grid
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)

    @torch.no_grad()
    def decode(
        self,
        preds: torch.Tensor,
        img_size: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        Decode raw network output into final detections.

        Args:
            preds: (B, S, S, B*5 + C) raw output
            img_size: (H, W) image size in pixels
            conf_thres: objectness × class score threshold
            iou_thres: IoU threshold for NMS
            max_det: maximum boxes to keep per image

        Returns:
            List of dicts, each with keys: boxes (Nx4), scores (N), labels (N)
        """
        B = preds.shape[0]
        H, W = img_size
        device = preds.device

        # 1) split into box predictions and class logits
        pred_boxes = preds[..., : self.B *
                           5].view(B, self.S, self.S, self.B, 5)
        pred_cls_logits = preds[..., self.B * 5:]  # (B, S, S, C)

        # 2) apply activations
        # sigmoid for x, y offsets and objectness
        tx = torch.sigmoid(pred_boxes[..., 0])
        ty = torch.sigmoid(pred_boxes[..., 1])
        tw = pred_boxes[..., 2].pow(2)  # sqrt→square
        th = pred_boxes[..., 3].pow(2)
        conf = torch.sigmoid(pred_boxes[..., 4])

        # softmax for class probabilities
        cls_probs = torch.softmax(pred_cls_logits, dim=-1)

        # 3) convert relative to absolute coords
        gx = self.grid_x.to(device)  # (1,S,S,1)
        gy = self.grid_y.to(device)
        cx = (tx + gx) * self.cell_size  # normalized center x
        cy = (ty + gy) * self.cell_size

        # now to pixel space
        x1 = (cx - tw / 2) * W
        y1 = (cy - th / 2) * H
        x2 = (cx + tw / 2) * W
        y2 = (cy + th / 2) * H
        boxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, S, S, B, 4)

        # 4) per‑image post‑processing: threshold & NMS
        dets_list: List[Dict[str, Any]] = []
        for b in range(B):
            # flatten all boxes for this image
            boxes_b = boxes_abs[b].reshape(-1, 4)
            obj_b = conf[b].reshape(-1, 1)  # objectness
            cls_b = cls_probs[b].reshape(self.S * self.S, self.C)
            # repeat class probs for each of the B boxes
            cls_b = cls_b.repeat_interleave(self.B, dim=0)  # (S*S*B, C)

            # scores = objectness * class probability
            scores_b = obj_b * cls_b

            # pick best class per box
            top_scores, top_labels = scores_b.max(dim=1)  # (N,)

            # threshold
            keep_mask = top_scores > conf_thres
            if not keep_mask.any():
                dets_list.append({"boxes": torch.empty((0, 4), device=device),
                                  "scores": torch.empty((0,), device=device),
                                  "labels": torch.empty((0,), dtype=torch.long, device=device)})
                continue

            boxes_keep = boxes_b[keep_mask]
            scores_keep = top_scores[keep_mask]
            labels_keep = top_labels[keep_mask]

            # class‑aware NMS
            keep_idx = tv_ops.batched_nms(
                boxes_keep, scores_keep, labels_keep, iou_thres)
            if keep_idx.numel() > max_det:
                keep_idx = keep_idx[:max_det]

            dets_list.append({
                "boxes":  boxes_keep[keep_idx],
                "scores": scores_keep[keep_idx],
                "labels": labels_keep[keep_idx],
            })

        return dets_list
