import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from capybara import draw_detection, dump_json, get_curdir, imwrite
from tabulate import tabulate
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .components import *  # noqa: F403  ← 依需求保留
from .utils import BaseMixin  # noqa: F401

DIR = get_curdir(__file__)


def _to_numpy_img(img: torch.Tensor) -> np.ndarray:
    """[0‑1] Tensor → uint8 HWC"""
    if img.is_cuda:
        img = img.detach().cpu()
    img = (img.clamp(0, 1) * 255).to(torch.uint8)
    return img.permute(1, 2, 0).contiguous().numpy().copy()


def _sanitize_boxes(
    boxes: torch.Tensor,
    img_wh: Tuple[int, int],
    min_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    將 xyxy 盒子做合法化處理
    1. 交換 (x1,x2)、(y1,y2) 確保 x1<=x2, y1<=y2
    2. clamp 至影像邊界 [0,W‑1]×[0,H‑1]
    3. 去除寬或高 < min_size 的框
    4. 回傳 (boxes_fix, keep_idx)

    參數
    -------
    boxes   : Tensor [N,4] (xyxy) ‑ 任意裝置、dtype 皆可
    img_wh  : (W,H)
    min_size: 最小寬/高（像素）

    回傳
    -------
    boxes_fix: [M,4]  — 已合法化、裝置同 boxes
    keep_idx : [M]    — 原 boxes 中被保留的 index
    """
    if boxes.numel() == 0:                 # 空 Tensor 直接回傳
        return boxes, boxes.new_empty((0,), dtype=torch.long)

    W, H = img_wh
    boxes = boxes.clone()

    # 1. 座標排序
    x1 = torch.min(boxes[:, 0], boxes[:, 2])
    y1 = torch.min(boxes[:, 1], boxes[:, 3])
    x2 = torch.max(boxes[:, 0], boxes[:, 2])
    y2 = torch.max(boxes[:, 1], boxes[:, 3])
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1, y1, x2, y2

    # 2. clamp 至影像範圍
    boxes[:, [0, 2]].clamp_(0, W - 1)
    boxes[:, [1, 3]].clamp_(0, H - 1)

    # 3. 篩掉寬 / 高 不足
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    keep = (wh[:, 0] >= min_size) & (wh[:, 1] >= min_size)
    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)

    return boxes[keep_idx], keep_idx


# -----------------------------------------------------------
# 主要模型
# -----------------------------------------------------------
class ObjectDetectionModel(BaseMixin, L.LightningModule):
    """
    通用目標偵測 LightningModule

    * 支援 backbone / neck / head 任意組合
    * 使用 torchmetrics.MeanAveragePrecision 產出 COCO‑style 指標
    * preview 功能：將 GT 與預測結果輸出成圖片 & JSON 方便檢閱
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.preview_batch = cfg.common.preview_batch
        self.apply_solver_config(
            cfg.optimizer, cfg.lr_scheduler)  # BaseMixin 提供

        # ---------------- Model 組裝 ----------------
        self.backbone, self.neck, self.head = nn.Identity(), nn.Identity(), nn.Identity()

        # Backbone
        bb_cfg = cfg.model.get("backbone", {})
        if bb_cfg:
            self.backbone = globals()[bb_cfg["name"]](
                **bb_cfg.get("options", {}))

        # Neck
        neck_cfg = cfg.model.get("neck", {})
        if neck_cfg:
            in_ch = getattr(self.backbone, "channels", [])
            neck_opts = {**neck_cfg.get("options", {}),
                         "in_channels_list": in_ch}
            self.neck = globals()[neck_cfg["name"]](**neck_opts)

        # Head
        head_cfg = cfg.model.get("head", {})
        if head_cfg:
            if hasattr(self.neck, "out_channels"):
                in_ch = [self.neck.out_channels] * \
                    cfg.common.num_feature_levels
            else:
                in_ch = getattr(self.backbone, "channels", [])
            head_opts = {**head_cfg.get("options", {}),
                         "in_channels_list": in_ch}
            self.head = globals()[head_cfg["name"]](**head_opts)

        # Loss
        loss_cfg = cfg.model.get("loss", {})
        self.criterion = globals()[loss_cfg["name"]](
            **loss_cfg.get("options", {}))

        # Metric：跨 epoch 重複使用，避免記憶體累加
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=False,
        )

    # -------------------------------------------------------
    # 前向
    # -------------------------------------------------------
    def forward(self, x: torch.Tensor):
        x_b = self.backbone(x)
        x_n = self.neck(x_b)
        preds = self.head(x_n)
        return preds, x_n, x_b

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------
    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        preds, *_ = self.forward(imgs)
        loss, loss_dict = self.criterion(preds, targets)

        if self.global_rank == 0 and not (batch_idx % self.preview_batch):
            self.preview(batch_idx, imgs, targets, preds, suffix="train")

        self.log_dict(
            {"lr": self.get_lr(), **loss_dict},
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )
        return loss

    # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch                       # imgs: [B,3,H,W]
        preds_raw, *_ = self.forward(imgs)
        B, _, H, W = imgs.shape

        dets_list = self.head.decode(               # 低閾值推論
            preds_raw,
            img_size=(H, W),
            conf_thres=0.001,
            iou_thres=0.65,
            max_det=300,
        )

        pred_dicts, tgt_dicts = [], []
        for det, tgt in zip(dets_list, targets):
            # ------------- Prediction -------------
            pred_dicts.append({
                "boxes":  det["boxes"].float().cpu(),
                "scores": det["scores"].float().cpu(),
                "labels": det["labels"].int().cpu(),
            })

            # ------------- Ground Truth -----------
            boxes_t = tgt["boxes"]
            if isinstance(boxes_t, np.ndarray):
                boxes_t = torch.as_tensor(boxes_t, device=imgs.device)

            if boxes_t.numel() and boxes_t.max() <= 1.0:        # ← 修正核心
                scale = torch.tensor([W, H, W, H], device=boxes_t.device)
                boxes_t = boxes_t * scale

            tgt_dicts.append({
                "boxes":  boxes_t.float().cpu(),
                "labels": tgt["labels"].int().cpu(),
            })

        # Metric 累積
        self.map_metric.update(pred_dicts, tgt_dicts)

        # 視覺化輸出
        if self.global_rank == 0 and not (batch_idx % self.preview_batch):
            self.preview_val(batch_idx, imgs, dets_list, targets, suffix="val")

    @torch.no_grad()
    def on_validation_epoch_end(self):
        """計算並輸出 COCO‑style 指標"""
        stats = self.map_metric.compute()      # Tensor dict
        self.map_metric.reset()                # 清空 for 下一個 epoch

        summary = {
            "mAP@0.50:0.95": stats["map"].item() * 100,
            "mAP@0.50":      stats["map_50"].item() * 100,
            "mAP@0.75":      stats["map_75"].item() * 100,
            "AR@1":          stats["mar_1"].item() * 100,
            "AR@10":         stats["mar_10"].item() * 100,
            "AR@100":        stats["mar_100"].item() * 100,
        }

        # Console 表格
        print("\n" + tabulate([summary],
                              headers="keys",
                              floatfmt=".2f",
                              tablefmt="psql") + "\n")

        # Lightning log（維持 0‑1 範圍）
        self.log_dict(
            {
                "val/map":    stats["map"],
                "val/map50":  stats["map_50"],
                "val/map75":  stats["map_75"],
                "val/ar1":    stats["mar_1"],
                "val/ar10":   stats["mar_10"],
                "val/ar100":  stats["mar_100"],
            },
            prog_bar=True,
            sync_dist=True,
        )

    # -------------------------------------------------------
    # 視覺化（train / val 共用）
    # -------------------------------------------------------
    @torch.no_grad()
    def preview(self, batch_idx, imgs, targets, preds, suffix="train"):
        """將 GT 與預測結果輸出成圖檔，方便肉眼檢查"""
        preview_dir = self.preview_dir / f"{suffix}_batch_{batch_idx}"
        preview_dir.mkdir(parents=True, exist_ok=True)

        B, _, H, W = imgs.shape
        dets_list = self.head.decode(
            preds, img_size=(H, W), conf_thres=0.25, iou_thres=0.45
        )

        for i in range(B):
            img_orig = _to_numpy_img(imgs[i])
            img_gt, img_pred = img_orig.copy(), img_orig.copy()

            # ---------- GT ----------
            tgt = targets[i]
            boxes_t = tgt["boxes"]
            if isinstance(boxes_t, torch.Tensor):
                if boxes_t.numel() and boxes_t.max() <= 1.0:
                    boxes_t = boxes_t * torch.tensor([W, H, W, H],
                                                     device=boxes_t.device)
                boxes_np = boxes_t.cpu().numpy()
            else:
                if boxes_t.size and boxes_t.max() <= 1.0:
                    boxes_np = boxes_t * np.array([W, H, W, H],
                                                  dtype=boxes_t.dtype)
                else:
                    boxes_np = boxes_t

            for box, lbl in zip(boxes_np, tgt["labels"]):
                img_gt = draw_detection(
                    img_gt, box, f"id:{int(lbl)}", thickness=2)

            # ---------- Prediction ----------
            det = dets_list[i]
            boxes_fix, good = _sanitize_boxes(det["boxes"], (W, H), min_size=2)
            scores_fix = det["scores"][good]
            labels_fix = det["labels"][good]

            for box, score, lbl in zip(boxes_fix, scores_fix, labels_fix):
                img_pred = draw_detection(
                    img_pred,
                    box.cpu().numpy(),
                    f"{int(lbl)}:{score:.2f}",
                    thickness=2,
                )

            # ---------- concat & save ----------
            out_path = preview_dir / f"sample_{i:03d}.jpg"
            imwrite(np.concatenate([img_gt, img_pred], axis=1), out_path)

            dump_json(
                dict(
                    gt=dict(boxes=boxes_np.tolist(),
                            labels=_to_list(tgt["labels"])),
                    pred=dict(boxes=det["boxes"].cpu().tolist(),
                              scores=det["scores"].cpu().tolist(),
                              labels=det["labels"].cpu().tolist()),
                ),
                preview_dir / f"sample_{i:03d}.json",
            )

    # 與 preview 邏輯相同，只是 decode 已經完成
    @torch.no_grad()
    def preview_val(self, batch_idx, imgs, dets_list, targets, suffix="val"):
        preview_dir = self.preview_dir / f"{suffix}_batch_{batch_idx}"
        preview_dir.mkdir(parents=True, exist_ok=True)

        B, _, H, W = imgs.shape
        for i in range(B):
            img_orig = _to_numpy_img(imgs[i])
            img_gt, img_pred = img_orig.copy(), img_orig.copy()

            # ---------- GT ----------
            tgt = targets[i]
            boxes_t = tgt["boxes"]
            if isinstance(boxes_t, torch.Tensor):
                if boxes_t.numel() and boxes_t.max() <= 1.0:
                    boxes_t = boxes_t * torch.tensor([W, H, W, H],
                                                     device=boxes_t.device)
                boxes_np = boxes_t.cpu().numpy()
            else:
                if boxes_t.size and boxes_t.max() <= 1.0:
                    boxes_np = boxes_t * np.array([W, H, W, H],
                                                  dtype=boxes_t.dtype)
                else:
                    boxes_np = boxes_t

            for box, lbl in zip(boxes_np, tgt["labels"]):
                img_gt = draw_detection(
                    img_gt, box, f"id:{int(lbl)}", thickness=2)

            # ---------- Prediction ----------
            det = dets_list[i]
            boxes_fix, good = _sanitize_boxes(det["boxes"], (W, H), min_size=2)
            scores_fix = det["scores"][good]
            labels_fix = det["labels"][good]

            for box, score, lbl in zip(boxes_fix, scores_fix, labels_fix):
                img_pred = draw_detection(
                    img_pred,
                    box.cpu().numpy(),
                    f"{int(lbl)}:{score:.2f}",
                    thickness=2,
                )

            # ---------- concat & save ----------
            out_path = preview_dir / f"sample_{i:03d}.jpg"
            imwrite(np.concatenate([img_gt, img_pred], axis=1), out_path)

            dump_json(
                dict(
                    gt=dict(boxes=boxes_np.tolist(),
                            labels=_to_list(tgt["labels"])),
                    pred=dict(boxes=det["boxes"].cpu().tolist(),
                              scores=det["scores"].cpu().tolist(),
                              labels=det["labels"].cpu().tolist()),
                ),
                preview_dir / f"sample_{i:03d}.json",
            )


# -----------------------------------------------------------
# 工具函式
# -----------------------------------------------------------
def _to_list(x):
    """Torch / ndarray → Python list"""
    if isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    return x.tolist()
