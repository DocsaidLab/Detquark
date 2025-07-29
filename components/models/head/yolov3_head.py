from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


class YOLOv3Head(nn.Module):
    """YOLOv3 detection head with multi-scale anchor-based predictions.

    Attributes:
        num_classes (int): Number of object classes.
        n_scales (int): Number of prediction scales.
        strides (List[int]): Downsampling strides for each scale.
        anchors (Tensor): Global anchor sizes in pixels, shape (A, 2).
        pred_layers (nn.ModuleList): Per-scale prediction convolutional layers.
        anchor_cells_s{i} (Tensor): Anchor sizes normalized by stride per scale.
        _grid_cache (OrderedDict): Cache for grid offsets keyed by (scale_id, Sy, Sx, device, dtype).
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int,
        anchors: List[Tuple[float, float]],
        anchor_masks: List[List[int]],
        strides: List[int],
        hid_channels: int = 1024,
    ) -> None:
        """Initializes YOLOv3 multi-scale detection head.

        Args:
            in_channels_list: List of input channel counts per scale.
            num_classes: Number of object classes.
            anchors: List of global anchor sizes in pixels [(w, h), ...].
            anchor_masks: List of anchor index subsets per scale.
            strides: List of downsampling strides per scale.
            hid_channels: Hidden channels for prediction conv layers.
        """
        super().__init__()

        if not (len(in_channels_list) == len(anchor_masks) == len(strides)):
            raise ValueError(
                "`in_channels_list`, `anchor_masks`, and `strides` must have equal length"
            )

        self.num_classes = num_classes
        self.n_scales = len(strides)
        self.strides = list(strides)

        anc = torch.as_tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anc)  # (A, 2)

        self.pred_layers = nn.ModuleList()
        for i, (c_in, mask, stride) in enumerate(zip(in_channels_list, anchor_masks, strides)):

            mask = list(mask)
            num_anchors = len(mask)
            self.register_buffer(
                f"anchor_cells_s{i}", self.anchors[mask] / stride
            )  # Normalize anchors by stride

            self.pred_layers.append(
                nn.Sequential(
                    nn.Conv2d(c_in, hid_channels, kernel_size=3,
                              padding=1, bias=False),
                    nn.BatchNorm2d(hid_channels),
                    nn.LeakyReLU(0.1, inplace=False),
                    nn.Conv2d(hid_channels, num_anchors *
                              (5 + num_classes), kernel_size=1),
                )
            )

        # Cache grid offsets for decoding, keyed by (scale_id, Sy, Sx, device, dtype)
        self._grid_cache: OrderedDict[
            Tuple[int, int, int, torch.device, torch.dtype],
            Tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass producing raw predictions for each scale.

        Args:
            features: List of feature maps, one per scale.

        Returns:
            List of raw prediction tensors shaped (B, Sy, Sx, A*(5+C)) per scale.
        """
        if len(features) != self.n_scales:
            raise ValueError(
                f"Expected {self.n_scales} feature maps, got {len(features)}"
            )

        outputs = []
        for feat, pred_layer in zip(features, self.pred_layers):
            p = pred_layer(feat)
            p = p.permute(0, 2, 3, 1).contiguous()
            outputs.append(p)

        return outputs

    def _get_grid(
        self,
        scale_id: int,
        Sy: int,
        Sx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or create grid offsets for box decoding at given scale and resolution.

        Args:
            scale_id: Index of prediction scale.
            Sy: Grid height.
            Sx: Grid width.
            device: Tensor device.
            dtype: Tensor dtype.

        Returns:
            Tuple of grid_x and grid_y tensors shaped (1, Sy, Sx, 1).
        """
        key = (scale_id, Sy, Sx, device, dtype)
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
        preds: List[torch.Tensor],
        img_size: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode raw multi-scale predictions into final detections.

        Args:
            preds: List of raw prediction tensors per scale.
            img_size: Original image size (height, width) in pixels.
            conf_thres: Confidence threshold for filtering detections.
            iou_thres: IoU threshold for non-maximum suppression.
            max_det: Maximum detections to keep per image.

        Returns:
            List of dictionaries (one per batch item) with keys:
                'boxes' (Tensor[N, 4]),
                'scores' (Tensor[N]),
                'labels' (Tensor[N]).
        """
        if len(preds) != self.n_scales:
            raise ValueError("Mismatch in number of prediction scales")

        B, *_ = preds[0].shape
        H, W = img_size
        device, dtype = preds[0].device, preds[0].dtype
        C = self.num_classes

        all_boxes, all_scores, all_labels = [], [], []
        for s, (pred, stride) in enumerate(zip(preds, self.strides)):
            B_s, Sy, Sx, _ = pred.shape
            a_i = pred.shape[-1] // (5 + C)

            pred = pred.view(B_s, Sy, Sx, a_i, 5 + C)
            tx, ty, tw, th, to = (
                pred[..., 0],
                pred[..., 1],
                pred[..., 2],
                pred[..., 3],
                pred[..., 4],
            )
            cls_logits = pred[..., 5:]

            cx = torch.sigmoid(tx)
            cy = torch.sigmoid(ty)
            bw = torch.exp(tw.clamp(max=8.0))
            bh = torch.exp(th.clamp(max=8.0))
            obj = torch.sigmoid(to)
            cls_prob = torch.softmax(cls_logits, dim=-1)

            grid_x, grid_y = self._get_grid(s, Sy, Sx, device, dtype)
            anc_cell = getattr(self, f"anchor_cells_s{s}")
            anc_w = anc_cell[:, 0].view(1, 1, 1, a_i)
            anc_h = anc_cell[:, 1].view(1, 1, 1, a_i)

            cx = (cx + grid_x) * stride
            cy = (cy + grid_y) * stride
            pw = anc_w * bw * stride
            ph = anc_h * bh * stride

            x1 = (cx - pw * 0.5).clamp(0, W - 1)
            y1 = (cy - ph * 0.5).clamp(0, H - 1)
            x2 = (cx + pw * 0.5).clamp(0, W - 1)
            y2 = (cy + ph * 0.5).clamp(0, H - 1)
            boxes = torch.stack((x1, y1, x2, y2), dim=-1)

            boxes = boxes.view(B, -1, 4)
            cls_prob = cls_prob.view(B, -1, C)
            obj = obj.view(B, -1, 1)
            scores = obj * cls_prob
            scores_max, labels = scores.max(dim=-1)

            all_boxes.append(boxes)
            all_scores.append(scores_max)
            all_labels.append(labels)

        boxes_cat = torch.cat(all_boxes, dim=1)
        scores_cat = torch.cat(all_scores, dim=1)
        labels_cat = torch.cat(all_labels, dim=1)

        results = []
        for b in range(B):
            mask = scores_cat[b] > conf_thres
            if not mask.any():
                results.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "scores": torch.empty(0, device=device),
                        "labels": torch.empty(0, dtype=torch.long, device=device),
                    }
                )
                continue

            boxes_b = boxes_cat[b][mask]
            scores_b = scores_cat[b][mask]
            labels_b = labels_cat[b][mask]

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
