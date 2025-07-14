from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


class YOLOv2Head(nn.Module):
    """YOLOv2 detection head with anchor-based bounding box prediction.

    Attributes:
        num_classes (int): Number of object classes.
        num_anchors (int): Number of anchors per grid cell.
        stride (int): Down-sampling factor from input image to feature map.
        anchors (Tensor): Anchor sizes in pixels, shape (A, 2).
        conv_pred (nn.Sequential): Convolutional predictor producing raw outputs.
        _grid_cache (Dict): Cache for grid offset tensors keyed by (Sy, Sx, device, dtype).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        anchors: List[Tuple[float, float]],
        stride: int,
        **kwargs,
    ):
        """Initializes the YOLOv2 detection head.

        Args:
            in_channels (int): Channels of the input feature map.
            num_classes (int): Number of object classes.
            anchors (List[Tuple[float, float]]): Anchor sizes in pixels as [(w, h), ...].
            stride (int): Down-sampling factor from input image to feature map.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.stride = stride

        anc = torch.as_tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anc)

        # Convolutional predictor producing raw detection output tensor
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
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute raw predictions from input features.

        Args:
            features (List[Tensor]): List of feature maps; uses the last one
                of shape (batch, C, Sy, Sx).

        Returns:
            Tensor: Raw prediction tensor of shape (batch, Sy, Sx, A*(5+C)).
        """
        x = features[-1]
        p = self.conv_pred(x)                    # (B, A*(5+C), Sy, Sx)
        p = p.permute(0, 2, 3, 1).contiguous()   # (B, Sy, Sx, A*(5+C))
        return p

    def _get_grid(
        self,
        Sy: int,
        Sx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or create grid offsets for decoding box centers.

        Args:
            Sy (int): Number of grid cells vertically.
            Sx (int): Number of grid cells horizontally.
            device (torch.device): Device of the tensors.
            dtype (torch.dtype): Data type of the tensors.

        Returns:
            Tuple[Tensor, Tensor]: grid_x and grid_y tensors shaped (1, Sy, Sx, 1).
        """
        key = (Sy, Sx, device, dtype)
        if key not in self._grid_cache:
            yv, xv = torch.meshgrid(
                torch.arange(Sy, device=device, dtype=dtype),
                torch.arange(Sx, device=device, dtype=dtype),
                indexing="ij",
            )
            grid_y = yv.unsqueeze(0).unsqueeze(-1)  # (1, Sy, Sx, 1)
            grid_x = xv.unsqueeze(0).unsqueeze(-1)  # (1, Sy, Sx, 1)
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
        """Decode raw network predictions into final detections.

        Args:
            preds (Tensor): Raw predictions of shape (B, Sy, Sx, A*(5+C)).
            img_size (Tuple[int, int]): (height, width) of the original image in pixels.
            conf_thres (float): Confidence threshold to filter detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
            max_det (int): Maximum number of detections to keep per image.

        Returns:
            List[Dict[str, Tensor]]: List of dicts per image containing keys
                'boxes' (Tensor[N,4]), 'scores' (Tensor[N]), and 'labels' (Tensor[N]).
        """
        B, Sy, Sx, _ = preds.shape
        H, W = img_size
        device, dtype = preds.device, preds.dtype
        A, C = self.num_anchors, self.num_classes

        # Reshape predictions to separate anchor and box components
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
        # 1. Apply activation functions to raw outputs
        # ------------------------------------------------------------------
        cx = torch.sigmoid(tx)                        # center x offsets
        cy = torch.sigmoid(ty)                        # center y offsets
        bw = torch.exp(tw.clamp(max=8.0))             # width scaling factor
        bh = torch.exp(th.clamp(max=8.0))             # height scaling factor
        obj = torch.sigmoid(to)                       # objectness score
        cls_prob = torch.softmax(cls_logits, dim=-1)  # class probabilities

        # ------------------------------------------------------------------
        # 2. Decode box center coordinates and sizes to image pixels
        # ------------------------------------------------------------------
        grid_x, grid_y = self._get_grid(Sy, Sx, device, dtype)
        anchor_w = self.anchors[:, 0].view(
            1, 1, 1, A).to(device=device, dtype=dtype)
        anchor_h = self.anchors[:, 1].view(
            1, 1, 1, A).to(device=device, dtype=dtype)

        cx = (cx + grid_x) * self.stride
        cy = (cy + grid_y) * self.stride
        pw = anchor_w * bw
        ph = anchor_h * bh

        # Convert center-size to corner coordinates and clamp to image bounds
        x1 = (cx - pw * 0.5).clamp(0, W - 1)
        y1 = (cy - ph * 0.5).clamp(0, H - 1)
        x2 = (cx + pw * 0.5).clamp(0, W - 1)
        y2 = (cy + ph * 0.5).clamp(0, H - 1)
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)  # (B, Sy, Sx, A, 4)

        # ------------------------------------------------------------------
        # 3. Flatten tensors for batch-wise NMS
        # ------------------------------------------------------------------
        boxes = boxes.view(B, -1, 4)       # (B, N, 4)
        cls_prob = cls_prob.view(B, -1, C)  # (B, N, C)
        obj = obj.view(B, -1, 1)            # (B, N, 1)
        scores = obj * cls_prob             # (B, N, C)
        scores_max, labels = scores.max(dim=-1)

        results = []
        for b in range(B):
            mask = scores_max[b] > conf_thres
            if not mask.any():
                results.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "scores": torch.empty((0,), device=device),
                        "labels": torch.empty((0,), dtype=torch.long, device=device),
                    }
                )
                continue

            boxes_b = boxes[b][mask]
            scores_b = scores_max[b][mask]
            labels_b = labels[b][mask]

            # Perform non-maximum suppression and limit detections
            keep = tv_ops.batched_nms(boxes_b, scores_b, labels_b, iou_thres)
            keep = keep[:max_det]

            results.append(
                {
                    "boxes": boxes_b[keep],
                    "scores": scores_b[keep],
                    "labels": labels_b[keep],
                }
            )

        return results
