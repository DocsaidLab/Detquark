from collections import OrderedDict
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


# ------------------------------------------------------------------ #
# NMS Helper (replaceable)
# ------------------------------------------------------------------ #
def default_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_thres: float,
    max_det: int,
) -> torch.Tensor:
    """Perform class-aware batched NMS, keeping top `max_det` indices.

    Args:
        boxes (torch.Tensor): Boxes tensor of shape ``[N, 4]`` in (x1, y1, x2, y2) format.
        scores (torch.Tensor): Scores tensor of shape ``[N]``.
        labels (torch.Tensor): Integer class labels tensor of shape ``[N]``.
        iou_thres (float): IoU threshold for NMS.
        max_det (int): Maximum number of detections to keep.

    Returns:
        torch.Tensor: Indices of kept boxes, with length ≤ ``max_det``.
    """
    keep = tv_ops.batched_nms(boxes, scores, labels, iou_thres)
    return keep[:max_det]


# ------------------------------------------------------------------ #
# YOLOv3 Detection Head
# ------------------------------------------------------------------ #
class YOLOv3Head(nn.Module):
    """Multi-scale YOLOv3 detection head with independent 1 x 1 conv predictors per scale.

    The output tensor shape per scale is ``(B, Sy, Sx, A*(5 + C))``, where
    ``A = len(anchor_masks[i])`` is the number of anchors at that scale.

    Attributes:
        num_classes (int): Number of object classes.
        num_scales (int): Number of feature scales.
        strides (List[int]): Down-sampling factors per scale.
        _nms (Callable): NMS function used during decoding.
        anchors (Tensor): Global anchors in pixels, shape ``[A, 2]``.
        conv_preds (ModuleList): 1 x 1 conv predictors per scale.
        _grid_cache_max (int): Maximum size of grid offset cache.
        _grid_cache (OrderedDict): LRU cache for grid offsets keyed by
            ``(Sy, Sx, device, dtype)``.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        num_classes: int,
        anchors: List[Tuple[float, float]],
        anchor_masks: List[List[int]],
        strides: List[int],
        *,
        nms_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, float, int], torch.Tensor
        ] = default_nms,
        grid_cache_size: int = 8,
    ) -> None:
        """Initialize YOLOv3Head.

        Args:
            in_channels_list (Sequence[int]): Channels of input feature maps per scale.
            num_classes (int): Number of object classes.
            anchors (List[Tuple[float, float]]): Global anchor list in pixels.
            anchor_masks (List[List[int]]): Anchor indices used per scale.
            strides (List[int]): Down-sampling factors per scale.
            nms_fn (Callable): NMS function with signature
                ``(boxes, scores, labels, iou_thr, max_det)``.
            grid_cache_size (int): Maximum LRU cache size for grid offsets.
        """
        super().__init__()

        if not (len(in_channels_list) == len(anchor_masks) == len(strides)):
            raise ValueError(
                "`in_channels_list`, `anchor_masks`, and `strides` must have equal lengths"
            )

        self.num_classes = int(num_classes)
        self.num_scales = len(strides)
        self.strides = list(strides)
        self._nms = nms_fn

        # Register global anchors as a buffer (shape: [A, 2])
        anc = torch.as_tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anc)

        # Build per-scale 1x1 conv predictors and register anchor buffers
        self.conv_preds = nn.ModuleList()
        for s, (in_ch, mask, stride) in enumerate(zip(in_channels_list, anchor_masks, strides)):
            mask = list(mask)
            if any(a >= len(anc) for a in mask):
                raise ValueError(
                    f"anchor_masks[{s}] contains out-of-range anchor indices")
            A = len(mask)

            # Conv layer outputs A * (5 + num_classes) channels for bounding box predictions
            conv = nn.Conv2d(in_ch, A * (5 + num_classes), kernel_size=1)
            self.conv_preds.append(conv)

            # Anchors per scale in pixel units and normalized cell units
            px = anc[mask]                      # shape: (A, 2), pixels
            # normalized to feature cell size
            cell = px / float(stride)
            self.register_buffer(f"anchors_px_s{s}", px)
            self.register_buffer(f"anchors_cell_s{s}", cell)

        # Initialize LRU cache for grid offsets
        self._grid_cache_max = int(grid_cache_size)
        self._grid_cache: OrderedDict[
            Tuple[int, int, torch.device, torch.dtype],
            Tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute raw predictions for each input feature map scale.

        Args:
            features (List[torch.Tensor]): List of feature maps, each with
                shape ``(B, C, Sy, Sx)``.

        Returns:
            List[torch.Tensor]: Raw prediction tensors per scale with shape
                ``(B, Sy, Sx, A*(5 + C))``.
        """
        if len(features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} feature maps, but got {len(features)}")

        outputs: List[torch.Tensor] = []
        for feat, conv in zip(features, self.conv_preds):
            p = conv(feat)                     # shape: (B, A*(5+C), Sy, Sx)
            p = p.permute(0, 2, 3, 1).contiguous()  # to (B, Sy, Sx, A*(5+C))
            outputs.append(p)
        return outputs

    def _get_grid(
        self,
        Sy: int,
        Sx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve or create LRU-cached grid offset tensors for feature map spatial coords.

        Args:
            Sy (int): Height of the feature map.
            Sx (int): Width of the feature map.
            device (torch.device): Target device.
            dtype (torch.dtype): Target tensor data type.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: grid_x and grid_y tensors
                with shape ``(1, Sy, Sx, 1)``.
        """
        key = (Sy, Sx, device, dtype)
        if key in self._grid_cache:
            self._grid_cache.move_to_end(key)
            return self._grid_cache[key]

        yv, xv = torch.meshgrid(
            torch.arange(Sy, device=device, dtype=dtype),
            torch.arange(Sx, device=device, dtype=dtype),
            indexing="ij",
        )
        grid_y = yv.unsqueeze(0).unsqueeze(-1)
        grid_x = xv.unsqueeze(0).unsqueeze(-1)

        # Enforce LRU cache size limit
        if len(self._grid_cache) >= self._grid_cache_max:
            self._grid_cache.popitem(last=False)
        self._grid_cache[key] = (grid_x, grid_y)
        return grid_x, grid_y

    @torch.no_grad()
    def decode(
        self,
        preds: List[torch.Tensor],
        img_size: Tuple[int, int],
        *,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        pre_nms_topk: int | None = 1000,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode raw model outputs into NMS-filtered final detections.

        Args:
            preds (List[torch.Tensor]): Raw prediction tensors per scale.
            img_size (Tuple[int, int]): Original image size (height, width).
            conf_thres (float): Confidence score threshold to filter candidates.
            iou_thres (float): IoU threshold used in NMS.
            max_det (int): Maximum number of detections per image post-NMS.
            pre_nms_topk (int | None): Optional pre-NMS top-K filtering threshold.

        Returns:
            List[Dict[str, torch.Tensor]]: List (length B) of dicts per batch item with keys:
                - "boxes": Tensor[N, 4] bounding boxes (x1, y1, x2, y2).
                - "scores": Tensor[N] confidence scores.
                - "labels": Tensor[N] class labels.
        """
        if len(preds) != self.num_scales:
            raise ValueError("Mismatch in number of prediction scales")

        H, W = img_size
        device, dtype = preds[0].device, preds[0].dtype
        B = preds[0].shape[0]
        C = self.num_classes

        all_boxes, all_scores, all_labels = [], [], []

        for s, p in enumerate(preds):
            B_s, Sy, Sx, _ = p.shape
            if B_s != B:
                raise ValueError(f"Batch size mismatch at scale {s}")

            A = p.shape[-1] // (5 + C)
            p = p.view(B, Sy, Sx, A, 5 + C)

            # Split predictions into components
            tx, ty, tw, th, to = p[..., 0], p[...,
                                              1], p[..., 2], p[..., 3], p[..., 4]
            cls_logits = p[..., 5:]

            # Apply activation functions
            bx = torch.sigmoid(tx)
            by = torch.sigmoid(ty)
            bw = torch.exp(tw.clamp(max=8.0))
            bh = torch.exp(th.clamp(max=8.0))
            obj = torch.sigmoid(to)
            cls_prob = torch.sigmoid(cls_logits)  # multi-label sigmoid

            # Retrieve grid offsets and anchors
            grid_x, grid_y = self._get_grid(Sy, Sx, device, dtype)
            anc_cell = getattr(self, f"anchors_cell_s{s}").to(dtype)  # (A, 2)
            anc_w = anc_cell[:, 0].view(1, 1, 1, A)
            anc_h = anc_cell[:, 1].view(1, 1, 1, A)
            stride = float(self.strides[s])

            # Decode bounding boxes to image coordinates
            cx = (bx + grid_x) * stride
            cy = (by + grid_y) * stride
            pw = anc_w * bw * stride
            ph = anc_h * bh * stride

            x1 = (cx - pw * 0.5).clamp(0, W - 1)
            y1 = (cy - ph * 0.5).clamp(0, H - 1)
            x2 = (cx + pw * 0.5).clamp(0, W - 1)
            y2 = (cy + ph * 0.5).clamp(0, H - 1)

            boxes = torch.stack((x1, y1, x2, y2), dim=-1).reshape(B, -1, 4)

            # Compute final scores (objectness  x  class probabilities)
            scores_all = (obj.unsqueeze(-1) * cls_prob).reshape(B, -1, C)

            # Get best class and corresponding score per box (Darknet style)
            best_scores, best_labels = scores_all.max(dim=-1)

            # Optional pre-NMS top-K filtering
            if pre_nms_topk is not None and best_scores.shape[1] > pre_nms_topk:
                vals, idxs = torch.topk(best_scores, pre_nms_topk, dim=1)
                gather_idx_boxes = idxs.unsqueeze(-1).expand(-1, -1, 4)
                boxes = torch.gather(boxes, 1, gather_idx_boxes)
                best_scores = vals
                best_labels = torch.gather(best_labels, 1, idxs)

            all_boxes.append(boxes)
            all_scores.append(best_scores)
            all_labels.append(best_labels)

        boxes_cat = torch.cat(all_boxes, dim=1)      # (B, N, 4)
        scores_cat = torch.cat(all_scores, dim=1)    # (B, N)
        labels_cat = torch.cat(all_labels, dim=1)    # (B, N)

        results: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            mask = scores_cat[b] > conf_thres
            if not mask.any():
                results.append({
                    "boxes":  torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                })
                continue

            b_boxes = boxes_cat[b][mask]
            b_scores = scores_cat[b][mask]
            b_labels = labels_cat[b][mask]

            keep = self._nms(b_boxes, b_scores, b_labels, iou_thres, max_det)

            results.append({
                "boxes":  b_boxes[keep].float(),
                "scores": b_scores[keep],
                "labels": b_labels[keep],
            })
        return results
