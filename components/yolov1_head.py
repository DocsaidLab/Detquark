from typing import Dict, List, Tuple

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
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        # feature extractor: conv → batchnorm → LeakyReLU
        # (paper did not include this extra conv+BN, remove here if you want exact reproduction)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        # fully-connected head: flatten → FC → LeakyReLU → Dropout(0.5) → final FC
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),
        )

        # pre-compute grid offsets for decoding
        gy, gx = torch.meshgrid(
            torch.arange(self.S), torch.arange(self.S), indexing="ij"
        )
        self.register_buffer("grid_x", gx.unsqueeze(0).unsqueeze(-1).float())
        self.register_buffer("grid_y", gy.unsqueeze(0).unsqueeze(-1).float())
        self.register_buffer("cell_size", torch.tensor(1.0 / self.S))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of intermediate features; use the last one (shape batchxC'xSxS)
        Returns:
            raw predictions tensor of shape (batch, S, S, B*5 + C)
        """
        x = features[-1]
        batch_size, _, Hf, Wf = x.shape
        assert Hf == self.S and Wf == self.S, (
            f"Expected feature map {self.S} x {self.S}, but got {Hf} x {Wf}"
        )

        x = self.conv(x)
        x = self.fc(x)
        return x.view(batch_size, self.S, self.S, self.B * 5 + self.C)

    @torch.no_grad()
    def decode(
        self,
        preds: torch.Tensor,
        img_size: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Decode raw network output into final detections per image.

        Args:
            preds: (batch, S, S, B*5 + C)
            img_size: (H, W) in pixels
            conf_thres: objectness threshold
            iou_thres: IoU threshold for NMS
            max_det: max boxes per image
        Returns:
            list of dicts with keys "boxes", "scores", "labels"
        """
        batch_size = preds.size(0)
        H, W = img_size
        device = preds.device

        # split predictions
        pred_boxes = preds[..., : self.B *
                           5].view(batch_size, self.S, self.S, self.B, 5)
        pred_cls_logits = preds[..., self.B * 5:]  # (batch, S, S, C)

        # activations
        tx = torch.sigmoid(pred_boxes[..., 0])
        ty = torch.sigmoid(pred_boxes[..., 1])
        raw_w = pred_boxes[..., 2]
        raw_h = pred_boxes[..., 3]
        # as in original paper: predict √w, √h → square to get w, h
        tw = raw_w.pow(2)
        th = raw_h.pow(2)
        conf = torch.sigmoid(pred_boxes[..., 4])

        cls_probs = torch.softmax(pred_cls_logits, dim=-1)

        # convert to absolute coords
        gx = self.grid_x.to(device)
        gy = self.grid_y.to(device)
        cs = self.cell_size.to(device)

        cx = (tx + gx) * cs
        cy = (ty + gy) * cs

        x1 = (cx - tw / 2) * W
        y1 = (cy - th / 2) * H
        x2 = (cx + tw / 2) * W
        y2 = (cy + th / 2) * H

        # clamp to image
        x1 = x1.clamp(0, W)
        y1 = y1.clamp(0, H)
        x2 = x2.clamp(0, W)
        y2 = y2.clamp(0, H)

        # (batch, S, S, B, 4)
        boxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)

        results: List[Dict[str, torch.Tensor]] = []
        num_cells = self.S * self.S
        for bi in range(batch_size):
            boxes_flat = boxes_abs[bi].reshape(-1, 4)
            conf_flat = conf[bi].reshape(-1)
            cls_cell = cls_probs[bi].reshape(num_cells, self.C)
            cls_flat = cls_cell.repeat_interleave(self.B, dim=0)

            mask = conf_flat > conf_thres
            if not mask.any():
                results.append({
                    "boxes": torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                })
                continue

            bboxes = boxes_flat[mask]
            confs = conf_flat[mask].unsqueeze(1)
            cls_p = cls_flat[mask]
            cls_scores, cls_labels = cls_p.max(dim=1)
            scores = (confs.squeeze(1) * cls_scores)

            keep = tv_ops.batched_nms(bboxes, scores, cls_labels, iou_thres)
            if keep.numel() > max_det:
                keep = keep[:max_det]

            results.append({
                "boxes":  bboxes[keep],
                "scores": scores[keep],
                "labels": cls_labels[keep],
            })

        return results
