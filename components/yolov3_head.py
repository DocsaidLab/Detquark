from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.ops as tv_ops


class YOLOv3Head(nn.Module):
    """
    Multi‑scale YOLOv3 detection head.

    Args:
        in_channels (List[int]): Channel dimensions of the incoming feature maps
            (one per scale, 深度通常為 [256, 512, 1024]).
        num_classes (int): Number of object classes (C).
        anchors (List[List[Tuple[float,float]]]): Anchor sizes (pixel) per scale,
            len(anchors) == len(in_channels); anchors[i] → stride[i].
        strides (List[int]): Down‑sampling factors for每個 feature map
            (e.g. [8, 16, 32]).
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        anchors: List[List[Tuple[float, float]]],
        strides: List[int],
        **kwargs,
    ):
        super().__init__()
        assert len(in_channels) == len(anchors) == len(strides), \
            "in_channels, anchors, strides 必須同長度。"
        self.num_classes = num_classes
        self.num_scales = len(in_channels)
        self.strides = strides

        # 針對每一層建立 anchors 與 anchor_cells
        self.register_buffer(
            "anchors",
            torch.cat([torch.tensor(a, dtype=torch.float32) for a in anchors]),
            persistent=False,
        )  # 方便導出 ONNX
        # List[Tensor(A_i,2)] → 儲存為普通屬性方便 decode
        self._anchor_cells: List[torch.Tensor] = [
            torch.as_tensor(a, dtype=torch.float32) / s
            for a, s in zip(anchors, strides)
        ]

        # per‑scale conv predictor
        self.conv_pred = nn.ModuleList()
        for in_ch, A in zip(in_channels, map(len, anchors)):
            self.conv_pred.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, 1024, 3, padding=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(
                        1024,
                        A * (5 + num_classes),  # tx ty tw th obj + C logits
                        kernel_size=1,
                    ),
                )
            )

        # grid cache per scale
        self._grid_cache: Dict[
            Tuple[int, int, int, torch.device, torch.dtype],
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}

    # -------------------------- forward -------------------------- #
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List[Tensor] with len == num_scales.
                      每張大小 (B, C_i, Sy_i, Sx_i).
        Returns:
            list of raw prediction tensors, each shape
            (B, Sy_i, Sx_i, A_i*(5+num_classes))
        """
        outputs: List[torch.Tensor] = []
        for i, (x, pred) in enumerate(zip(features, self.conv_pred)):
            B, _, Sy, Sx = x.shape
            p = pred(x)                                # (B, A*(5+C), Sy, Sx)
            p = p.permute(0, 2, 3, 1).contiguous()
            outputs.append(
                p.view(B, Sy, Sx, -1)                  # -1 = A_i*(5+C)
            )
        return outputs

    # ------------------------- helpers --------------------------- #
    def _get_grid(
        self,
        idx: int,
        Sy: int,
        Sx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        取得第 idx 個尺度的 grid tensor。
        """
        key = (idx, Sy, Sx, device, dtype)
        if key not in self._grid_cache:
            yv = torch.arange(Sy, device=device, dtype=dtype).view(Sy, 1)
            xv = torch.arange(Sx, device=device, dtype=dtype).view(1, Sx)
            grid_y = yv.expand(Sy, Sx).view(1, Sy, Sx, 1)
            grid_x = xv.expand(Sy, Sx).view(1, Sy, Sx, 1)
            self._grid_cache[key] = (grid_x, grid_y)
        return self._grid_cache[key]

    # --------------------------- decode -------------------------- #
    @torch.no_grad()
    def decode(
        self,
        preds: List[torch.Tensor],
        img_size: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            preds: List of raw tensors from ``forward``.
            img_size: (H, W) 原始影像大小。
        Returns:
            List[Dict[str, Tensor]] 與 YOLOv2Head 相同格式。
        """
        H, W = img_size
        device, dtype = preds[0].device, preds[0].dtype
        C = self.num_classes

        all_boxes: List[torch.Tensor] = []
        all_obj:   List[torch.Tensor] = []
        all_cls:   List[torch.Tensor] = []

        for idx, p in enumerate(preds):                 # 逐尺度解碼
            B, Sy, Sx, _ = p.shape
            A = p.shape[-1] // (5 + C)

            p = p.view(B, Sy, Sx, A, 5 + C)
            tx, ty, tw, th, to = (
                p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]
            )
            cls_logits = p[..., 5:]

            # activations
            bx = torch.sigmoid(tx)
            by = torch.sigmoid(ty)
            bw = torch.exp(tw.clamp(max=8))
            bh = torch.exp(th.clamp(max=8))
            obj = torch.sigmoid(to)                     # (B, Sy, Sx, A)
            cls_conf = torch.sigmoid(cls_logits)        # (B, Sy, Sx, A, C)

            # decode to pixel coords
            grid_x, grid_y = self._get_grid(idx, Sy, Sx, device, dtype)
            anc = self._anchor_cells[idx].to(device=device, dtype=dtype)
            anc_w = anc[:, 0].view(1, 1, 1, A)
            anc_h = anc[:, 1].view(1, 1, 1, A)
            stride = self.strides[idx]

            cx = (bx + grid_x) * stride
            cy = (by + grid_y) * stride
            pw = anc_w * bw * stride
            ph = anc_h * bh * stride

            x1 = (cx - pw * 0.5).clamp(0, W - 1)
            y1 = (cy - ph * 0.5).clamp(0, H - 1)
            x2 = (cx + pw * 0.5).clamp(0, W - 1)
            y2 = (cy + ph * 0.5).clamp(0, H - 1)
            boxes = torch.stack((x1, y1, x2, y2), -1)   # (B, Sy, Sx, A, 4)

            all_boxes.append(boxes.view(B, -1, 4))
            # keep dim for broadcast
            all_obj.append(obj.view(B, -1, 1))
            all_cls.append(cls_conf.view(B, -1, C))

        # concat 三個尺度 → (B, N_all, •)
        boxes = torch.cat(all_boxes, dim=1)
        obj = torch.cat(all_obj,   dim=1)
        cls_c = torch.cat(all_cls,   dim=1)

        # 最終 per‑class scores = obj * cls_conf
        scores_full = obj * cls_c                       # (B, N, C)

        results: List[Dict[str, torch.Tensor]] = []
        N_total = boxes.size(1)
        for b in range(B):
            boxes_b = boxes[b]

            # 依類別分開做 NMS，torchvision batched_nms 需要 flat vectors
            scores_list, labels_list, idxs_list = [], [], []
            for c in range(C):
                sc = scores_full[b, :, c]
                mask = sc > conf_thres
                if mask.any():
                    keep = tv_ops.nms(
                        boxes_b[mask], sc[mask], iou_thres
                    )[:max_det]  # per‑class top‑K
                    idx_global = torch.nonzero(
                        mask, as_tuple=False).squeeze(1)[keep]
                    scores_list.append(sc[idx_global])
                    idxs_list.append(idx_global)
                    labels_list.append(
                        torch.full_like(idx_global, c, dtype=torch.long)
                    )

            if scores_list:
                scores_cat = torch.cat(scores_list)
                idxs_cat = torch.cat(idxs_list)
                labels_cat = torch.cat(labels_list)

                # 再以 score 排序保留 top max_det
                order = scores_cat.argsort(descending=True)[:max_det]
                results.append(
                    {
                        "boxes":  boxes_b[idxs_cat[order]].float(),
                        "scores": scores_cat[order],
                        "labels": labels_cat[order],
                    }
                )
            else:
                results.append(
                    {
                        "boxes":  torch.empty((0, 4), device=device),
                        "scores": torch.empty((0,), device=device),
                        "labels": torch.empty((0,), dtype=torch.long, device=device),
                    }
                )
        return results
