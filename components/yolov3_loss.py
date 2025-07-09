from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ↓ 建議你比照 v2 的 utils，自行實作 build_targets_yolov3
from .utils import build_targets_yolov3


class YOLOv3Loss(nn.Module):
    """
    YOLOv3 multi‑scale loss (x‑y‑w‑h + objectness + multi‑label class).

    **主要差異**
    1. 支援多層特徵圖；`anchors` 應為「List[ List[(w,h)] ]」。
    2. 分類採 *per‑class sigmoid* → `BCEWithLogitsLoss`。
    3. 依照官方實踐，x / y 使用 sigmoid，再與 target 平方差；w / h 直接
       對 log‑encoded 值做 MSE（同 v2）。
    4. 可透過 `lambda_coord / lambda_noobj` 調整 box‑損失與背景抑制權重。

    Attributes
    ----------
    num_classes : int
        物件類別數 C。
    anchors : List[Tensor[A_i, 2]]
        每層 anchors (pixel)；供 debug / 匹配器使用。
    lambda_coord, lambda_noobj : float
        box‑loss 與 no‑obj BCE 的權重。
    ignore_iou_thr : float
        負樣本忽略門檻。
    mse, bce : nn.Module
        分別為 MSE 與 BCEWithLogits（sum reduction）。
    """

    def __init__(
        self,
        anchors: List[List[Tuple[float, float]]],  # per-scale anchors
        num_classes: int,
        img_dim: int = 416,
        lambda_coord: float = 1.0,
        lambda_noobj: float = 1.0,
        ignore_iou_thr: float = 0.7,
    ):
        super().__init__()
        # 儲存 per‑scale anchors 為 buffer，方便 ONNX 匯出
        self.anchor_buffers: nn.ParameterList = nn.ParameterList()
        for anc in anchors:
            t = torch.tensor(anc, dtype=torch.float32)
            self.anchor_buffers.append(nn.Parameter(t, requires_grad=False))

        self.num_classes = num_classes
        self.img_dim = img_dim
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.ignore_iou_thr = ignore_iou_thr

        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")

    # --------------------------------------------------------------------- #
    #                                Forward                                #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        preds: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        preds : List[Tensor]
            來自 `YOLOv3Head.forward`，長度 = #scales，
            每張 (B, Sy, Sx, A_i*(5 + C)).
        targets : List[dict]
            與 v2 相同：{'boxes': (N,4) xyxy norm, 'labels': (N,)}

        Returns
        -------
        total_loss : Tensor
            綜合 loss（scalar）。
        stats : Dict[str, Tensor]
            個別項目與樣本統計，鍵：loss, coord, obj, noobj, cls, pos, neg。
        """
        device = preds[0].device
        C = self.num_classes

        # 累積器（先存 raw sum，最後除以樣本數）
        coord_sum = obj_sum = noobj_sum = cls_sum = torch.tensor(
            0., device=device
        )
        pos_total = neg_total = torch.tensor(0., device=device)

        # -------------------- per‑scale 計算 -------------------- #
        for p, anc_buf in zip(preds, self.anchor_buffers):
            B, Sy, Sx, _ = p.shape
            A = anc_buf.size(0)
            # 轉成 (B, Sy, Sx, A, 5+C)
            p = p.view(B, Sy, Sx, A, 5 + C)
            tx, ty, tw, th, to = (
                p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]
            )
            tcls_logit = p[..., 5:]

            # ----------- 建立 ground‑truth target / mask ----------- #
            (
                tgt_xywh,          # (B,Sy,Sx,A,4) encoded
                tgt_conf,          # (B,Sy,Sx,A)   0/1
                tgt_cls,           # (B,Sy,Sx,A,C) one‑hot
                obj_mask,          # bool
                noobj_mask,        # bool
            ) = build_targets_yolov3(
                targets=targets,
                anchors=anc_buf,    # Tensor[A,2]
                S=(Sy, Sx),
                img_dim=self.img_dim,
                num_classes=C,
                ignore_iou_thr=self.ignore_iou_thr,
                device=device,
            )

            # 安全避免除 0
            n_pos = obj_mask.sum().clamp(min=1).float()
            n_neg = noobj_mask.sum().clamp(min=1).float()
            pos_total += n_pos
            neg_total += n_neg

            # ----------------------- Losses ----------------------- #
            # 1) box
            px = torch.sigmoid(tx[obj_mask])
            py = torch.sigmoid(ty[obj_mask])
            pw = tw[obj_mask]
            ph = th[obj_mask]
            tgt_xywh_pos = tgt_xywh[obj_mask]  # (n_pos,4)

            coord_sum += (
                self.mse(px, tgt_xywh_pos[:, 0])
                + self.mse(py, tgt_xywh_pos[:, 1])
                + self.mse(pw, tgt_xywh_pos[:, 2])
                + self.mse(ph, tgt_xywh_pos[:, 3])
            ) * self.lambda_coord

            # 2) obj / no‑obj
            obj_sum += self.bce(to[obj_mask], tgt_conf[obj_mask])
            noobj_sum += (
                self.bce(to[noobj_mask], tgt_conf[noobj_mask])
                * self.lambda_noobj
            )

            # 3) class (multi‑label BCE)
            cls_sum += self.bce(
                tcls_logit[obj_mask], tgt_cls[obj_mask]
            )

        # -------------------- 標準化與回傳 -------------------- #
        coord = coord_sum / pos_total
        obj = obj_sum / pos_total
        noobj = noobj_sum / neg_total
        cls = cls_sum / pos_total

        total_loss = coord + obj + noobj + cls
        stats = {
            "loss":  total_loss.detach(),
            "coord": coord.detach(),
            "obj":   obj.detach(),
            "noobj": noobj.detach(),
            "cls":   cls.detach(),
            "pos":   pos_total.detach(),
            "neg":   neg_total.detach(),
        }
        return total_loss, stats
