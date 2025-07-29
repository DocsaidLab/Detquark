import inspect
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from capybara import draw_detection, dump_json, get_curdir, imwrite
from tabulate import tabulate
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .components import *
from .utils import BaseMixin

# Directory of this file
DIR = get_curdir(__file__)


def _to_numpy_img(img: torch.Tensor) -> np.ndarray:
    """
    Convert a [0-1] CHW float32 tensor to HWC uint8 numpy array.

    Args:
        img (torch.Tensor): Tensor with shape (3, H, W) and values in [0,1].

    Returns:
        np.ndarray: Image array with shape (H, W, 3) dtype uint8.
    """
    if img.is_cuda:
        img = img.detach().cpu()
    img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    # Convert CHW to HWC
    return img.permute(1, 2, 0).contiguous().numpy().copy()


def _to_list(x: Union[torch.Tensor, np.ndarray]) -> list:
    """
    Convert a torch.Tensor or numpy.ndarray to a Python list.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    return x.tolist()


def _sanitize_boxes(
    boxes: torch.Tensor,
    img_wh: Tuple[int, int],
    min_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sanitize bounding boxes in (x1, y1, x2, y2) format.
    1. Ensure x1 <= x2 and y1 <= y2 by swapping if needed.
    2. Clamp coordinates to image boundaries [0, W-1] and [0, H-1].
    3. Remove boxes with width or height smaller than min_size.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4).
        img_wh (Tuple[int, int]): (width, height) of the image.
        min_size (int): Minimum width/height in pixels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            boxes_fix: Sanitized boxes of shape (M,4).
            keep_idx: Indices of kept boxes in the original tensor.
    """
    if boxes.numel() == 0:
        return boxes, boxes.new_empty((0,), dtype=torch.long)

    W, H = img_wh
    boxes = boxes.clone().float()

    # 1. Sort coordinates so that x1<=x2 and y1<=y2
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1, y1, x2, y2

    # 2. Clamp to image bounds
    boxes[:, [0, 2]].clamp_(0, W - 1)
    boxes[:, [1, 3]].clamp_(0, H - 1)

    # 3. Filter out small boxes
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths >= min_size) & (heights >= min_size)
    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    boxes_fix = boxes[keep_idx]

    return boxes_fix, keep_idx


class ObjectDetectionModel(BaseMixin, L.LightningModule):
    """
    Generic object detection LightningModule supporting arbitrary backbone, neck, and head.

    Features:
    - Configurable via `cfg` dict.
    - COCO-style metrics via MeanAveragePrecision.
    - Preview and validation visualization utilities.
    """

    REGISTRY: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str, component: Type[nn.Module]) -> None:
        """Adds a class to the component registry."""
        cls.REGISTRY[name] = component

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.preview_batch = cfg.common.preview_batch
        self.apply_solver_config(cfg.optimizer, cfg.lr_scheduler)

        # Build model components
        self.backbone = nn.Identity()
        self.neck = nn.Identity()
        self.head = nn.Identity()

        # Build components
        self.backbone = self._build_component(cfg["model"]["backbone"])
        self.neck = (
            self._build_component(
                cfg["model"]["neck"],
                extra_kwargs={"in_channels_list": self._infer_neck_channels()},
            )
            if cfg.model.get("neck", {})
            else nn.Identity()
        )
        self.head = self._build_component(cfg["model"]["head"])
        self.criterion = self._build_component(cfg["model"]["loss"])

        # Metrics
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=False
        )

    def _infer_neck_channels(self) -> List[int]:
        """Finds backbone output channels to feed into the neck."""
        # Preferred explicit attribute
        if hasattr(self.backbone, "feature_info"):
            return [f["num_chs"] for f in self.backbone.feature_info]

        # Fallback attribute
        if hasattr(self.backbone, "channels"):
            return list(self.backbone.channels)

        raise RuntimeError(
            "Cannot infer neck `in_channels_list`. "
            "Ensure backbone exposes `feature_info` or `channels`."
        )

    def _build_component(
        self,
        cfg_section: Dict[str, Any],
        *,
        extra_kwargs: Dict[str, Any] | None = None,
    ) -> nn.Module:
        """Instantiates a component from the registry plus any dynamic kwargs."""
        name: str = cfg_section["name"]
        options: Dict[str, Any] = {**cfg_section.get("options", {})}
        if extra_kwargs:
            options = {**options, **extra_kwargs}

        if name not in self.REGISTRY:
            # Fallback to globals for backward compatibility
            cls = globals().get(name)
            if inspect.isclass(cls) and issubclass(cls, nn.Module):
                self.register(name, cls)
            else:
                raise ValueError(f"Component {name!r} is not registered.")

        return self.REGISTRY[name](**options)

    def forward(self, x: torch.Tensor):
        """
        Forward pass: backbone -> neck -> head.

        Returns:
            preds: Raw predictions from head.
            neck_feats: Output of neck.
            backbone_feats: Output of backbone.
        """
        backbone_feats = self.backbone(x)
        neck_feats = self.neck(backbone_feats)
        preds = self.head(neck_feats)
        return preds, neck_feats, backbone_feats

    def training_step(
        self,
        batch: Tuple[torch.Tensor, List[Dict[str, Any]]],
        batch_idx: int
    ):
        imgs, targets = batch
        preds, *_ = self.forward(imgs)

        # Compute loss
        loss, loss_dict = self.criterion(preds, targets)

        # Preview occasionally
        if self.global_rank == 0 and batch_idx % self.preview_batch == 0:
            self.preview(batch_idx, imgs, targets, preds, suffix="train")

        # Log learning rate and loss metrics
        self.log_dict(
            {"lr": self.get_lr(), **loss_dict},
            prog_bar=True,
            on_step=True,
            sync_dist=True
        )
        return loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, List[Dict[str, Any]]],
        batch_idx: int
    ):
        imgs, targets = batch

        # Outputs
        # preds_raw[0].shape = [B, H/8, W/8, 255]
        # preds_raw[1].shape = [B, H/16, W/16, 255]
        # preds_raw[2].shape = [B, H/32, W/32, 255]
        preds, *_ = self.forward(imgs)
        _, _, H, W = imgs.shape

        # Decode predictions at low threshold
        dets_list = self.head.decode(
            preds,
            img_size=(H, W),
            conf_thres=0.001,
            iou_thres=0.65,
            max_det=300,
        )

        pred_dicts, tgt_dicts = [], []
        for det, tgt in zip(dets_list, targets):
            # Prepare prediction dict
            pred_dicts.append({
                "boxes": det["boxes"].float().cpu(),
                "scores": det["scores"].float().cpu(),
                "labels": det["labels"].int().cpu(),
            })

            # Prepare ground truth dict
            boxes_t = tgt["boxes"]
            if isinstance(boxes_t, np.ndarray):
                boxes_t = torch.as_tensor(boxes_t, device=imgs.device)

            # Scale normalized boxes to pixels
            if boxes_t.numel() > 0 and boxes_t.max() <= 1.0:
                scale = torch.tensor(
                    [W, H, W, H], device=boxes_t.device, dtype=boxes_t.dtype)
                boxes_t = boxes_t.float() * scale

            tgt_dicts.append({
                "boxes": boxes_t.float().cpu(),
                "labels": tgt["labels"].int().cpu(),
            })

        # Accumulate metrics
        self.map_metric.update(pred_dicts, tgt_dicts)

        # Validation preview
        if self.global_rank == 0 and batch_idx % self.preview_batch == 0:
            self.preview_val(batch_idx, imgs, dets_list, targets, suffix="val")

    @torch.no_grad()
    def on_validation_epoch_end(self):
        """
        Compute and log COCO-style metrics at end of validation epoch.
        """
        stats = self.map_metric.compute()
        self.map_metric.reset()

        summary = {
            "mAP@0.50:0.95": stats["map"].item() * 100,
            "mAP@0.50": stats["map_50"].item() * 100,
            "mAP@0.75": stats["map_75"].item() * 100,
            "AR@1": stats["mar_1"].item() * 100,
            "AR@10": stats["mar_10"].item() * 100,
            "AR@100": stats["mar_100"].item() * 100,
        }

        # Print table to console
        print("\n\n\n" + tabulate([summary], headers="keys",
              floatfmt=".2f", tablefmt="psql") + "\n\n")

        # Log to Lightning (values in [0,1])
        self.log_dict(
            {f"val/{k}": v for k, v in zip(
                ["map", "map50", "map75", "ar1", "ar10", "ar100"],
                [stats["map"], stats["map_50"], stats["map_75"],
                    stats["mar_1"], stats["mar_10"], stats["mar_100"]]
            )},
            prog_bar=False,
            sync_dist=True,
        )

    @torch.no_grad()
    def preview(
        self,
        batch_idx: int,
        imgs: torch.Tensor,
        targets: List[Dict[str, Any]],
        preds: Any,
        suffix: str = "train",
    ):
        """
        Save side-by-side visualization of ground truth and predictions.
        """
        preview_dir = self.preview_dir / f"{suffix}_batch_{batch_idx}"
        preview_dir.mkdir(parents=True, exist_ok=True)

        B, _, H, W = imgs.shape
        dets_list = self.head.decode(
            preds,
            img_size=(H, W),
            conf_thres=0.25,
            iou_thres=0.45
        )

        for i in range(B):
            img_orig = _to_numpy_img(imgs[i])
            img_gt, img_pred = img_orig.copy(), img_orig.copy()

            # Draw ground truth boxes
            tgt = targets[i]
            boxes_t = tgt["boxes"]
            if isinstance(boxes_t, torch.Tensor):
                if boxes_t.numel() > 0 and boxes_t.max() <= 1.0:
                    scale = torch.tensor(
                        [W, H, W, H], device=boxes_t.device, dtype=boxes_t.dtype)
                    boxes_np = (boxes_t.float() * scale).cpu().numpy()
                else:
                    boxes_np = boxes_t.cpu().numpy()
            else:
                boxes_np = boxes_t if not (boxes_t.size and boxes_t.max() <= 1.0) else (
                    boxes_t * np.array([W, H, W, H], dtype=boxes_t.dtype))

            for box, lbl in zip(boxes_np, tgt["labels"]):
                img_gt = draw_detection(
                    img_gt, box, f"id:{int(lbl)}", thickness=2)

            # Draw predicted boxes
            det = dets_list[i]
            boxes_fix, keep_idx = _sanitize_boxes(
                det["boxes"], (W, H), min_size=2)
            scores_fix = det["scores"][keep_idx]
            labels_fix = det["labels"][keep_idx]

            for box, score, lbl in zip(boxes_fix, scores_fix, labels_fix):
                img_pred = draw_detection(
                    img_pred,
                    box.cpu().numpy(),
                    f"id:{int(lbl)}",
                    np.round(score.float().cpu().numpy(), 2),
                    thickness=2,
                )

            # Concatenate and save image
            out_img = np.concatenate([img_gt, img_pred], axis=1)
            imwrite(out_img, preview_dir / f"sample_{i:03d}.jpg")

            # Save JSON
            dump_json(
                {
                    "gt": {"boxes": boxes_np.tolist(), "labels": _to_list(tgt["labels"])},
                    "pred": {
                        "boxes": det["boxes"].cpu().tolist(),
                        "scores": det["scores"].cpu().tolist(),
                        "labels": det["labels"].cpu().tolist(),
                    },
                },
                preview_dir / f"sample_{i:03d}.json",
            )

    @torch.no_grad()
    def preview_val(
        self,
        batch_idx: int,
        imgs: torch.Tensor,
        dets_list: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, Any]],
        suffix: str = "val",
    ):
        """
        Same as `preview`, but using already-decoded detections.
        """
        preview_dir = self.preview_dir / f"{suffix}_batch_{batch_idx}"
        preview_dir.mkdir(parents=True, exist_ok=True)

        B, _, H, W = imgs.shape
        for i in range(B):
            img_orig = _to_numpy_img(imgs[i])
            img_gt, img_pred = img_orig.copy(), img_orig.copy()

            # Ground truth
            tgt = targets[i]
            boxes_t = tgt["boxes"]
            if isinstance(boxes_t, torch.Tensor):
                if boxes_t.numel() > 0 and boxes_t.max() <= 1.0:
                    scale = torch.tensor(
                        [W, H, W, H], device=boxes_t.device, dtype=boxes_t.dtype)
                    boxes_np = (boxes_t.float() * scale).cpu().numpy()
                else:
                    boxes_np = boxes_t.cpu().numpy()
            else:
                boxes_np = boxes_t if not (boxes_t.size and boxes_t.max() <= 1.0) else (
                    boxes_t * np.array([W, H, W, H], dtype=boxes_t.dtype))

            for box, lbl in zip(boxes_np, tgt["labels"]):
                img_gt = draw_detection(
                    img_gt, box, f"id:{int(lbl)}", thickness=2)

            # Predictions
            det = dets_list[i]
            boxes_fix, keep_idx = _sanitize_boxes(
                det["boxes"], (W, H), min_size=2)
            scores_fix = det["scores"][keep_idx]
            labels_fix = det["labels"][keep_idx]

            for box, score, lbl in zip(boxes_fix, scores_fix, labels_fix):
                img_pred = draw_detection(
                    img_pred,
                    box.cpu().numpy(),
                    f"id:{int(lbl)}",
                    np.round(score.float().cpu().numpy(), 2),
                    thickness=2,
                )

            out_img = np.concatenate([img_gt, img_pred], axis=1)
            imwrite(out_img, preview_dir / f"sample_{i:03d}.jpg")

            dump_json(
                {
                    "gt": {"boxes": boxes_np.tolist(), "labels": _to_list(tgt["labels"])},
                    "pred": {
                        "boxes": det["boxes"].cpu().tolist(),
                        "scores": det["scores"].cpu().tolist(),
                        "labels": det["labels"].cpu().tolist(),
                    },
                },
                preview_dir / f"sample_{i:03d}.json",
            )
