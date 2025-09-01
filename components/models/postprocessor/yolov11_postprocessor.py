from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


class YOLOv11Postprocessor(nn.Module):
    """Post-processes raw outputs from **YOLOv11Head** into final detections.

    The module converts the head's per-level logits into pixel-space bounding
    boxes, per-box class scores, and class indices, then applies confidence
    filtering and *batched* NMS.

    Attributes:
        reg_max: Integer ``R`` — number of DFL bins (e.g. ``16``).
        strides: Feature-map strides for each detection level.
        num_classes: Number of object classes (``C``).
        conf_thres: Score threshold used to filter low-confidence boxes before
            NMS.
        iou_thres: IoU threshold for class-aware batched NMS.
        max_det: Maximum detections returned per image.
        _grid_cache: Lazy cache → ``(1, 2, H, W)`` tensors storing
            feature-map coordinates in *(x, y)* order.
    """

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        reg_max: int,
        strides: Sequence[int],
        num_classes: int,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.strides = list(strides)
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self._grid_cache: dict[
            Tuple[int, int, torch.device], torch.Tensor] = {}

    # --------------------------------------------------------------------- #
    def _get_grid(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a cached coordinate grid.

        The grid represents feature-map center coordinates
        ``(0 … W-1, 0 … H-1)`` in *(x, y)* order.

        Args:
            h: Feature-map height.
            w: Feature-map width.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tensor of shape ``(1, 2, H, W)`` containing *(x, y)* centers.
        """
        key = (h, w, device, dtype)
        if key not in self._grid_cache:
            yv, xv = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            self._grid_cache[key] = torch.stack((xv, yv), dim=0).unsqueeze(0)
        return self._grid_cache[key]

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def forward(
        self,
        preds: Sequence[torch.Tensor],
        dfl_layer: nn.Module,
        img_size: Tuple[int, int],
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert multi-level logits to per-image detections.

        Args:
            preds: List of level tensors with shape
                ``(B, 4·R + C, H_l, W_l)``.
            dfl_layer: Pre-instantiated **DFLIntegral** layer used to convert
                discrete logits into ``l, t, r, b`` distances.
            img_size: Original image size ``(H_img, W_img)``.

        Returns:
            A list with length ``B``. Each element is a dict:
            ``{"boxes": (N, 4), "scores": (N,), "labels": (N,)}`` where
            boxes are in **XYXY** pixel coordinates.
        """
        h_img, w_img = img_size
        device, dtype = preds[0].device, preds[0].dtype
        batch_size = preds[0].shape[0]

        all_boxes, all_scores, all_labels = [], [], []

        # ---------------- Level-wise decoding ---------------- #
        for lvl, pred in enumerate(preds):
            bs, _, h, w = pred.shape
            stride = self.strides[lvl]
            bos_end = 4 * self.reg_max  # slice index for box logits

            # (1) Split logits ------------------------------------------------
            box_logits = pred[:, :bos_end, ...]       # (B, 4R, H, W)
            cls_logits = pred[:, bos_end:, ...]       # (B, C,  H, W)

            # (2) DFL integral → l, t, r, b -----------------------------------
            ltrb = dfl_layer(box_logits.flatten(2)).view(bs, 4, h, w)

            # (3) Convert to XYXY in pixel space --------------------------------
            grid = self._get_grid(h, w, device, dtype) + 0.5  # center
            xy_center = grid * stride  # (1,2,H,W)

            left, top, right, bottom = (ltrb * stride).unbind(1)
            x1 = (xy_center[:, 0] - left).flatten(1)
            y1 = (xy_center[:, 1] - top).flatten(1)
            x2 = (xy_center[:, 0] + right).flatten(1)
            y2 = (xy_center[:, 1] + bottom).flatten(1)
            # (B, HW, 4)
            boxes = torch.stack((x1, y1, x2, y2), dim=2)

            # (4) Class scores -------------------------------------------------
            scores = cls_logits.sigmoid().flatten(2).transpose(1, 2)  # (B, HW, C)
            scores_max, labels = scores.max(dim=-1)                   # (B, HW)

            all_boxes.append(boxes)
            all_scores.append(scores_max)
            all_labels.append(labels)

        # --------------- Concatenate all levels --------------- #
        boxes_cat = torch.cat(all_boxes, dim=1)    # (B, N, 4)
        scores_cat = torch.cat(all_scores, dim=1)  # (B, N)
        labels_cat = torch.cat(all_labels, dim=1)  # (B, N)

        # ---------------- Per-image filtering ---------------- #
        results: List[Dict[str, torch.Tensor]] = []
        for b in range(batch_size):
            keep_mask = scores_cat[b] > self.conf_thres
            if not keep_mask.any():
                results.append(
                    {
                        "boxes": torch.empty(0, 4, device=device),
                        "scores": torch.empty(0, device=device),
                        "labels": torch.empty(0, dtype=torch.long, device=device),
                    }
                )
                continue

            boxes_b = boxes_cat[b][keep_mask]
            scores_b = scores_cat[b][keep_mask]
            labels_b = labels_cat[b][keep_mask]

            # Clamp to image bounds ------------------------------------------
            boxes_b[:, [0, 2]].clamp_(0, w_img - 1)
            boxes_b[:, [1, 3]].clamp_(0, h_img - 1)

            # Batched NMS ------------------------------------------------------
            boxes_b = boxes_b.float()
            scores_b = scores_b.float()
            keep = tv_ops.batched_nms(
                boxes_b, scores_b, labels_b, self.iou_thres)
            keep = keep[: self.max_det]

            results.append(
                {
                    "boxes": boxes_b[keep],
                    "scores": scores_b[keep],
                    "labels": labels_b[keep],
                }
            )

        return results
