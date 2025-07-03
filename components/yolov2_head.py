from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


class YOLOv2Head(nn.Module):
    """
    Fully-convolutional anchor-based detection head (YOLOv2-style).

    Output tensor shape
    -------------------
    (B, Sy, Sx, A * (5 + C)), where the 5 anchor terms are
    [tx, ty, tw, th, to].

    Args
    ----
    in_channels   : Number of channels in the incoming feature map.
    num_classes   : Number of object classes (C).
    anchors       : List of (w, h) anchor sizes in input-image pixels.
    stride        : Total down-sampling factor of the feature map
                    relative to the input image.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        anchors: List[Tuple[float, float]],
        stride: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.stride = stride

        anc = torch.as_tensor(anchors, dtype=torch.float32)
        # raw pixel anchors (useful for debug / export)
        self.register_buffer("anchors", anc, persistent=False)
        # anchor sizes expressed in *cell* units for decoding
        self.register_buffer("anchor_cells", anc / stride, persistent=False)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                1024,
                self.num_anchors * (5 + self.num_classes),
                kernel_size=1,
            ),
        )

        # {(Sy, Sx, device, dtype): (grid_x, grid_y)}
        self._grid_cache: Dict[
            Tuple[int, int, torch.device,
                  torch.dtype], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, C, Sy, Sx) feature map from the backbone / neck.

        Returns
        -------
        Raw prediction tensor shaped (B, Sy, Sx, A * (5 + C)).
        """
        B, _, Sy, Sx = x.shape
        p = self.conv_pred(x)                       # (B, A*(5+C), Sy, Sx)
        p = p.permute(0, 2, 3, 1).contiguous()      # (B, Sy, Sx, A*(5+C))
        return p.view(B, Sy, Sx, self.num_anchors * (5 + self.num_classes))

    # ------------------------------------------------------------ grid utilities
    def _get_grid(
        self, Sy: int, Sx: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create (or retrieve cached) grid offsets of shape
        (1, Sy, Sx, 1) in x and y directions.
        """
        key = (Sy, Sx, device, dtype)
        if key not in self._grid_cache:
            yv = torch.arange(Sy, device=device, dtype=dtype).view(Sy, 1)
            xv = torch.arange(Sx, device=device, dtype=dtype).view(1, Sx)
            grid_y = yv.expand(Sy, Sx).view(1, Sy, Sx, 1)
            grid_x = xv.expand(Sy, Sx).view(1, Sy, Sx, 1)
            self._grid_cache[key] = (grid_x, grid_y)
        return self._grid_cache[key]

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def decode(
        self,
        preds: torch.Tensor,
        img_size: Tuple[int, int],
        *,
        conf_thr: float = 0.25,
        iou_thr: float = 0.45,
        max_det: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert raw predictions to NMS-filtered detections.

        Parameters
        ----------
        preds     : (B, Sy, Sx, A*(5+C)) – raw network output.
        img_size  : (H, W) of the original input image in pixels.
        conf_thr  : Minimum score (obj * class) to consider a proposal.
        iou_thr   : IoU threshold for batched NMS.
        max_det   : Max detections returned per image.

        Returns
        -------
        List of length B with dicts:
            - boxes  : (N, 4)  xyxy coordinates in pixels (float32)
            - scores : (N,)
            - labels : (N,)   class indices (int64)
        """
        B, Sy, Sx, _ = preds.shape
        H, W = img_size
        device, dtype = preds.device, preds.dtype
        A, C = self.num_anchors, self.num_classes

        preds = preds.view(B, Sy, Sx, A, 5 + C)
        tx = preds[..., 0]
        ty = preds[..., 1]
        tw = preds[..., 2]
        th = preds[..., 3]
        to = preds[..., 4]
        cls_logits = preds[..., 5:]

        bx = torch.sigmoid(tx)
        by = torch.sigmoid(ty)
        # guard overflow
        bw = torch.exp(tw.clamp(max=8.0))
        bh = torch.exp(th.clamp(max=8.0))
        obj = torch.sigmoid(to)
        cls_prob = torch.softmax(cls_logits, dim=-1)            # (B,Sy,Sx,A,C)

        # Generate grid cell offsets
        grid_x, grid_y = self._get_grid(Sy, Sx, device, dtype)  # (1,Sy,Sx,1)
        anc_w = self.anchor_cells[:, 0].view(1, 1, 1, A).to(dtype)
        anc_h = self.anchor_cells[:, 1].view(1, 1, 1, A).to(dtype)

        cx = (bx + grid_x) * self.stride
        cy = (by + grid_y) * self.stride
        pw = anc_w * bw * self.stride
        ph = anc_h * bh * self.stride

        # xywh ➜ xyxy (clamped to image bounds)
        x1 = (cx - pw * 0.5).clamp(0, W - 1)
        y1 = (cy - ph * 0.5).clamp(0, H - 1)
        x2 = (cx + pw * 0.5).clamp(0, W - 1)
        y2 = (cy + ph * 0.5).clamp(0, H - 1)
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)           # (B,Sy,Sx,A,4)

        # flatten proposals
        boxes = boxes.view(B, -1, 4)                            # (B, N, 4)
        obj = obj.view(B, -1)                                 # (B, N)
        cls_p = cls_prob.view(B, -1, C)                         # (B, N, C)

        results: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            # best-class selection per box
            cls_scores, cls_labels = cls_p[b].max(dim=-1)       # (N,)
            scores = obj[b] * cls_scores                        # fused score

            # thresholding
            keep_mask = scores > conf_thr
            if not keep_mask.any():
                results.append({
                    "boxes":  torch.empty(0, 4, device=device),
                    "scores": torch.empty(0, device=device),
                    "labels": torch.empty(0, dtype=torch.long, device=device),
                })
                continue

            boxes_b = boxes[b][keep_mask]
            scores_b = scores[keep_mask]
            labels_b = cls_labels[keep_mask]

            # class-wise NMS
            keep_idx = tv_ops.batched_nms(boxes_b, scores_b, labels_b, iou_thr)
            keep_idx = keep_idx[:max_det]

            results.append({
                "boxes":  boxes_b[keep_idx].float(),
                "scores": scores_b[keep_idx],
                "labels": labels_b[keep_idx],
            })

        return results
