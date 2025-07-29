from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


class YOLOv3Postprocessor(nn.Module):
    """
    Decode raw YOLOv3 predictions into final bounding boxes, scores, and labels.
    Supports construction via either anchor_cells or anchors + anchor_masks + strides.
    """

    def __init__(
        self,
        num_classes: int,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        *,
        anchor_cells: Optional[List[torch.Tensor]] = None,
        anchors: Optional[List[List[float]]] = None,
        anchor_masks: Optional[List[List[int]]] = None,
        strides: Optional[List[int]] = None,
    ):
        """
        Args:
            num_classes: Number of object classes.
            conf_thres: Confidence threshold.
            iou_thres: NMS IoU threshold.
            max_det: Maximum detections per image.
            anchor_cells: List of per-scale anchor sizes normalized by stride.
            anchors: Global anchor sizes (pixels).
            anchor_masks: Anchor groupings per feature level.
            strides: Downsampling strides per feature level.
        """
        super().__init__()

        if anchor_cells is None:
            assert anchors is not None and anchor_masks is not None and strides is not None, \
                "If anchor_cells is not provided, anchors, anchor_masks, and strides must be."
            all_anchors = torch.tensor(anchors, dtype=torch.float32)
            anchor_cells = [all_anchors[mask] / stride for mask,
                            stride in zip(anchor_masks, strides)]

        self.anchor_cells = anchor_cells
        self.strides = strides
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self._grid_cache = {}

    def _get_grid(self, s: int, Sy: int, Sx: int, device, dtype):
        key = (s, Sy, Sx, device, dtype)
        if key not in self._grid_cache:
            yv, xv = torch.meshgrid(
                torch.arange(Sy, device=device, dtype=dtype),
                torch.arange(Sx, device=device, dtype=dtype),
                indexing="ij",
            )
            grid_y = yv.unsqueeze(0).unsqueeze(-1)
            grid_x = xv.unsqueeze(0).unsqueeze(-1)
            self._grid_cache[key] = (grid_x, grid_y)
        return self._grid_cache[key]

    @torch.no_grad()
    def forward(
        self,
        preds: List[torch.Tensor],
        img_size: Tuple[int, int],
    ) -> List[Dict[str, torch.Tensor]]:
        H, W = img_size
        device, dtype = preds[0].device, preds[0].dtype
        C = self.num_classes

        all_boxes, all_scores, all_labels = [], [], []
        for s, (pred, stride, anchor_cell) in enumerate(zip(preds, self.strides, self.anchor_cells)):
            B, Sy, Sx, _ = pred.shape
            A = pred.shape[-1] // (5 + C)
            pred = pred.view(B, Sy, Sx, A, 5 + C)

            tx, ty, tw, th, to = pred[..., 0], pred[...,
                                                    1], pred[..., 2], pred[..., 3], pred[..., 4]
            cls_logits = pred[..., 5:]

            cx = torch.sigmoid(tx)
            cy = torch.sigmoid(ty)
            bw = torch.exp(tw.clamp(max=8.0))
            bh = torch.exp(th.clamp(max=8.0))
            obj = torch.sigmoid(to)
            cls_prob = torch.softmax(cls_logits, dim=-1)

            grid_x, grid_y = self._get_grid(s, Sy, Sx, device, dtype)

            anchor_cell = anchor_cell.to(
                device=device, dtype=dtype)  # ensure match
            anc_w = anchor_cell[:, 0].view(1, 1, 1, A)
            anc_h = anchor_cell[:, 1].view(1, 1, 1, A)

            cx = (cx + grid_x) * stride
            cy = (cy + grid_y) * stride
            pw = anc_w * bw * stride
            ph = anc_h * bh * stride

            x1 = (cx - pw * 0.5).clamp(0, W - 1)
            y1 = (cy - ph * 0.5).clamp(0, H - 1)
            x2 = (cx + pw * 0.5).clamp(0, W - 1)
            y2 = (cy + ph * 0.5).clamp(0, H - 1)

            boxes = torch.stack((x1, y1, x2, y2), dim=-1).view(B, -1, 4)
            scores = (obj.unsqueeze(-1) * cls_prob).view(B, -1, C)
            scores_max, labels = scores.max(dim=-1)

            all_boxes.append(boxes)
            all_scores.append(scores_max)
            all_labels.append(labels)

        boxes_cat = torch.cat(all_boxes, dim=1)
        scores_cat = torch.cat(all_scores, dim=1)
        labels_cat = torch.cat(all_labels, dim=1)

        results = []
        for b in range(boxes_cat.shape[0]):
            mask = scores_cat[b] > self.conf_thres
            if not mask.any():
                results.append({"boxes": torch.empty((0, 4), device=device),
                                "scores": torch.empty(0, device=device),
                                "labels": torch.empty(0, dtype=torch.long, device=device)})
                continue

            boxes_b = boxes_cat[b][mask]
            scores_b = scores_cat[b][mask]
            labels_b = labels_cat[b][mask]

            keep = tv_ops.batched_nms(boxes_b, scores_b, labels_b, self.iou_thres)[
                :self.max_det]
            results.append({
                "boxes": boxes_b[keep],
                "scores": scores_b[keep],
                "labels": labels_b[keep],
            })

        return results
