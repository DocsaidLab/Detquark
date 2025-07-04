from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


class YOLOv1Head(nn.Module):
    """
    YOLOv1 detection head for object detection.

    Attributes:
        S (int): Number of grid cells along each image dimension.
        B (int): Number of bounding boxes predicted per grid cell.
        C (int): Number of object classes.
        conv (nn.Module): Convolutional feature extractor.
        fc (nn.Module): Fully connected layers to produce raw predictions.
        grid_x (Tensor): X-coordinate offsets for each grid cell.
        grid_y (Tensor): Y-coordinate offsets for each grid cell.
        cell_size (Tensor): Normalized size of each grid cell (1/S).
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
        """
        Initialize YOLOv1Head.

        Args:
            in_channels (int): Channels of input feature map.
            num_classes (int): Number of object classes (C).
            num_bboxes (int): Boxes per cell (B). Default is 2.
            grid_size (int): Grid size (S). Default is 7.
            dropout (float): Dropout probability after hidden layer. Default is 0.5.
        """
        super().__init__()
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        # Extract and refine spatial features before prediction
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        # Map extracted features to raw prediction vector
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),
        )

        # Precompute grid offsets and cell size for decoding
        gy, gx = torch.meshgrid(
            torch.arange(self.S), torch.arange(self.S), indexing="ij"
        )
        self.register_buffer("grid_x", gx.unsqueeze(0).unsqueeze(-1).float())
        self.register_buffer("grid_y", gy.unsqueeze(0).unsqueeze(-1).float())
        self.register_buffer("cell_size", torch.tensor(1.0 / self.S))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Run features through the detection head to produce raw predictions.

        Args:
            features (List[Tensor]): List of intermediate feature maps; uses the last
                one of shape (batch, C', S, S).

        Returns:
            Tensor: Raw predictions of shape (batch, S, S, B*5 + C).

        Raises:
            AssertionError: If feature map spatial size != S.
        """
        x = features[-1]
        batch_size, _, Hf, Wf = x.shape
        # Ensure feature map matches expected grid size
        assert Hf == self.S and Wf == self.S, (
            f"Expected feature map {self.S}×{self.S}, got {Hf}×{Wf}"
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
        Decode raw predictions into final detections.

        Args:
            preds (Tensor): Raw output (batch, S, S, B*5 + C).
            img_size (Tuple[int, int]): (height, width) of original image in pixels.
            conf_thres (float): Threshold for objectness score.
            iou_thres (float): IoU threshold for non-max suppression.
            max_det (int): Maximum detections per image.

        Returns:
            List[Dict[str, Tensor]]: Each dict has keys 'boxes', 'scores', 'labels'.
        """
        batch_size = preds.size(0)
        H, W = img_size
        device = preds.device

        # ------------------------------------------------------------------
        # 1. Split predictions into box parameters and class logits
        # ------------------------------------------------------------------
        pred_boxes = preds[..., : self.B *
                           5].view(batch_size, self.S, self.S, self.B, 5)
        pred_cls_logits = preds[..., self.B * 5:]

        # ------------------------------------------------------------------
        # 2. Activate predictions
        # ------------------------------------------------------------------
        tx = torch.sigmoid(pred_boxes[..., 0])        # center x offset
        ty = torch.sigmoid(pred_boxes[..., 1])        # center y offset
        tw = pred_boxes[..., 2].pow(2)                # width
        th = pred_boxes[..., 3].pow(2)                # height
        conf = torch.sigmoid(pred_boxes[..., 4])      # objectness
        cls_probs = torch.softmax(pred_cls_logits, dim=-1)

        # ------------------------------------------------------------------
        # 3. Compute absolute box coordinates
        # ------------------------------------------------------------------
        gx = self.grid_x.to(device)
        gy = self.grid_y.to(device)
        cs = self.cell_size.to(device)
        cx = (tx + gx) * cs
        cy = (ty + gy) * cs

        x1 = (cx - tw / 2) * W
        y1 = (cy - th / 2) * H
        x2 = (cx + tw / 2) * W
        y2 = (cy + th / 2) * H

        # Clamp coordinates to image boundaries
        x1 = x1.clamp(0, W)
        y1 = y1.clamp(0, H)
        x2 = x2.clamp(0, W)
        y2 = y2.clamp(0, H)

        # (batch, S, S, B, 4)
        boxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)

        results: List[Dict[str, torch.Tensor]] = []
        num_cells = self.S * self.S

        # ------------------------------------------------------------------
        # 4. Post-process each image: filter, score, NMS
        # ------------------------------------------------------------------
        for bi in range(batch_size):
            # Flatten grid and anchor dims
            boxes_flat = boxes_abs[bi].reshape(-1, 4)
            conf_flat = conf[bi].reshape(-1)
            cls_flat = (
                cls_probs[bi]
                .reshape(num_cells, self.C)
                .repeat_interleave(self.B, dim=0)
            )

            # Filter low-confidence boxes
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
            scores = confs.squeeze(1) * cls_scores

            # Apply non-maximum suppression, limit to max_det
            keep = tv_ops.batched_nms(bboxes, scores, cls_labels, iou_thres)
            if keep.numel() > max_det:
                keep = keep[:max_det]

            results.append({
                "boxes":  bboxes[keep],
                "scores": scores[keep],
                "labels": cls_labels[keep],
            })

        return results
