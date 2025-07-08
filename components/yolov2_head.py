from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


class YOLOv2Head(nn.Module):
    """
    Anchor-based YOLOv2 detection head with fully convolutional prediction.

    Attributes:
        num_classes (int): Number of object classes.
        num_anchors (int): Number of anchors per grid cell.
        stride (int): Down-sampling factor from input image to feature map.
        anchors (Tensor): Anchor sizes in pixels, shape (A, 2).
        anchor_cells (Tensor): Anchor sizes in feature-map cell units, shape (A, 2).
        conv_pred (nn.Sequential): Convs mapping features to raw prediction tensor.
        _grid_cache (Dict): Cache for grid offset tensors keyed by (Sy, Sx, device, dtype).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        anchors: List[Tuple[float, float]],
        stride: int,
        **kwargs
    ):
        """
        Initialize YOLOv2Head.

        Args:
            in_channels (int): Number of channels in the incoming feature map.
            num_classes (int): Number of object classes (C).
            anchors (List[Tuple[float, float]]): Anchor sizes in pixels.
            stride (int): Down-sampling factor from input image to feature map.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.stride = stride

        anc = torch.as_tensor(anchors, dtype=torch.float32)
        # Raw pixel anchors for debugging or exporting
        self.register_buffer("anchors", anc, persistent=False)
        # Anchor sizes in cell units for decoding
        self.register_buffer("anchor_cells", anc / stride, persistent=False)

        # Convolutional predictor: conv3x3 → BN → LeakyReLU → conv1x1 to outputs
        self.conv_pred = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(
                1024,
                self.num_anchors * (5 + self.num_classes),
                kernel_size=1,
            ),
        )

        # Cache for grid offsets to avoid recomputation
        self._grid_cache: Dict[
            Tuple[int, int, torch.device, torch.dtype],
            Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Produce raw prediction tensor from feature maps.

        Args:
            features (List[Tensor]): List of feature maps; uses the last one
                of shape (B, C, Sy, Sx).

        Returns:
            Tensor: Raw predictions of shape (B, Sy, Sx, A*(5 + C)).
        """
        x = features[-1]
        B, _, Sy, Sx = x.shape

        # Apply conv layers and reshape to (B, Sy, Sx, A*(5+C))
        p = self.conv_pred(x)                      # (B, A*(5+C), Sy, Sx)
        p = p.permute(0, 2, 3, 1).contiguous()     # (B, Sy, Sx, A*(5+C))
        return p.view(B, Sy, Sx, self.num_anchors * (5 + self.num_classes))

    def _get_grid(
        self,
        Sy: int,
        Sx: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create or retrieve cached grid cell offsets.

        Args:
            Sy (int): Number of grid rows.
            Sx (int): Number of grid columns.
            device (torch.device): Device for the grid tensors.
            dtype (torch.dtype): Data type for the grid tensors.

        Returns:
            Tuple[Tensor, Tensor]: grid_x and grid_y each of shape (1, Sy, Sx, 1).
        """
        key = (Sy, Sx, device, dtype)
        if key not in self._grid_cache:
            # Build grid coordinates
            yv = torch.arange(Sy, device=device, dtype=dtype).view(Sy, 1)
            xv = torch.arange(Sx, device=device, dtype=dtype).view(1, Sx)
            grid_y = yv.expand(Sy, Sx).view(1, Sy, Sx, 1)
            grid_x = xv.expand(Sy, Sx).view(1, Sy, Sx, 1)
            self._grid_cache[key] = (grid_x, grid_y)
        return self._grid_cache[key]

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
        Convert raw network output into NMS-filtered detections.

        Args:
            preds (Tensor): Raw predictions of shape (B, Sy, Sx, A*(5+C)).
            img_size (Tuple[int, int]): Original image size (H, W) in pixels.
            conf_thres (float): Confidence threshold for filtering. Default 0.25.
            iou_thres (float): IoU threshold for batched NMS. Default 0.45.
            max_det (int): Maximum detections per image. Default 300.

        Returns:
            List[Dict[str, Tensor]]: For each batch item, dictionary with:
                'boxes'  (Tensor[N, 4]): xyxy coords in pixels,
                'scores' (Tensor[N]): confidence scores,
                'labels' (Tensor[N]): class indices.
        """
        B, Sy, Sx, _ = preds.shape
        H, W = img_size
        device, dtype = preds.device, preds.dtype
        A, C = self.num_anchors, self.num_classes

        # Reshape to (B, Sy, Sx, A, 5+C)
        preds = preds.view(B, Sy, Sx, A, 5 + C)
        tx, ty, tw, th, to = (
            preds[..., 0],
            preds[..., 1],
            preds[..., 2],
            preds[..., 3],
            preds[..., 4],
        )
        cls_logits = preds[..., 5:]

        # ------------------------------------------------------------------
        # 1. Apply activations
        # ------------------------------------------------------------------
        # center x in cell units
        bx = torch.sigmoid(tx)
        # center y in cell units
        by = torch.sigmoid(ty)
        bw = torch.exp(tw.clamp(max=8.0))                # width scaling
        bh = torch.exp(th.clamp(max=8.0))                # height scaling
        obj = torch.sigmoid(to)                          # objectness score
        cls_prob = torch.softmax(cls_logits, dim=-1)     # class probabilities

        # ------------------------------------------------------------------
        # 2. Decode box centers and sizes to pixel coordinates
        # ------------------------------------------------------------------
        grid_x, grid_y = self._get_grid(Sy, Sx, device, dtype)
        anc_w = self.anchor_cells[:, 0].view(1, 1, 1, A)
        anc_h = self.anchor_cells[:, 1].view(1, 1, 1, A)

        cx = (bx + grid_x) * self.stride
        cy = (by + grid_y) * self.stride
        pw = anc_w * bw * self.stride
        ph = anc_h * bh * self.stride

        # Convert to corner coordinates and clamp
        x1 = (cx - pw * 0.5).clamp(0, W - 1)
        y1 = (cy - ph * 0.5).clamp(0, H - 1)
        x2 = (cx + pw * 0.5).clamp(0, W - 1)
        y2 = (cy + ph * 0.5).clamp(0, H - 1)
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)    # (B, Sy, Sx, A, 4)

        # ------------------------------------------------------------------
        # 3. Flatten for NMS
        # ------------------------------------------------------------------
        boxes = boxes.view(B, -1, 4)                     # (B, N, 4)
        scores = (obj.view(B, -1) *                        # objectness × class score
                  cls_prob.view(B, -1, C).max(dim=-1).values)

        cls_labels = cls_prob.view(B, -1, C).argmax(dim=-1)

        results: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            mask = scores[b] > conf_thres
            if not mask.any():
                # No detections above threshold
                results.append({
                    "boxes":  torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                })
                continue

            boxes_b = boxes[b][mask]
            scores_b = scores[b][mask]
            labels_b = cls_labels[b][mask]

            # Class-wise non-maximum suppression
            keep = tv_ops.batched_nms(boxes_b, scores_b, labels_b, iou_thres)
            keep = keep[:max_det]

            results.append({
                "boxes":  boxes_b[keep].float(),
                "scores": scores_b[keep],
                "labels": labels_b[keep],
            })

        return results
