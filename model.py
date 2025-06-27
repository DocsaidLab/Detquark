from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from capybara import (Polygon, Polygons, draw_polygon, draw_text, dump_json,
                      get_curdir, imbinarize, imresize, imwrite, jaccard_index)
from tabulate import tabulate
from torchmetrics import Accuracy

from .components import *
from .utils import BaseMixin

DIR = get_curdir(__file__)


class ObjectDetectionModel(BaseMixin, L.LightningModule):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.preview_batch = cfg.common.preview_batch
        self.apply_solver_config(cfg.optimizer, cfg.lr_scheduler)

        # Setup model
        cfg_model = cfg['model']
        self.backbone = nn.Identity()
        self.neck = nn.Identity()
        self.head = nn.Identity()

        # Build backbone
        backbone_cfg = cfg.model.get("backbone") or {}
        bb_name = backbone_cfg.get("name")
        bb_opts = backbone_cfg.get("options", {})
        self.backbone = globals()[bb_name](**bb_opts)

        # Build neck
        neck_cfg = cfg.model.get("neck") or {}
        neck_name = neck_cfg.get("name")
        neck_opts = neck_cfg.get("options", {})
        if neck_name:
            # ensure in_channels_list based on backbone
            channels = getattr(self.backbone, "channels", [])
            neck_opts.setdefault("in_channels_list", channels)
            self.neck = globals()[neck_name](**neck_opts)
        else:
            self.neck = nn.Identity()

        # Build head
        head_cfg = cfg.model.get("head") or {}
        head_name = head_cfg.get("name")
        head_opts = head_cfg.get("options", {})
        if head_name:
            # infer in_channels_list for head
            if hasattr(self.neck, "out_channels"):
                head_opts.setdefault("in_channels_list", [
                                     self.neck.out_channels] * cfg.common.num_feature_levels)
            else:
                head_opts.setdefault("in_channels_list", getattr(
                    self.backbone, "channels", []))
            head_opts.setdefault(
                "max_text_length", cfg.common.get("max_text_length", 0))
            self.head = globals()[head_name](**head_opts)
        else:
            self.head = nn.Identity()

        # Setup loss function

        # for validation
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_backbone = self.backbone(x)
        x_neck = self.neck(x_backbone)
        outputs = self.head(x_neck)
        return outputs, x_neck, x_backbone

    def training_step(self, batch, batch_idx):

        imgs, targets = batch
        preds, x_neck, x_backbone = self(imgs)
        breakpoint()

        # priview
        # if not (batch_idx % self.preview_batch) and self.global_rank == 0:
        #     self.preview(
        #         batch_idx, imgs,
        #         text_gts, logits
        #     )

        # self.log_dict(
        #     {
        #         'lr': self.get_lr(),
        #         'loss': loss,
        #         'acc': acc,
        #     },
        #     prog_bar=True,
        #     on_step=True,
        #     sync_dist=True,
        # )

        # return loss

    # def validation_step(self, batch, batch_idx):
    #     imgs, _, text_gts = batch

    #     logits = self.forward(imgs)

    #     _text_gts = self.text_enc(text_gts)[0]
    #     _text_gts = torch.LongTensor(_text_gts).to(device=logits.device)

    #     # priview
    #     # if not (batch_idx % self.preview_batch) and self.global_rank == 0:
    #     self.preview(
    #         batch_idx, imgs,
    #         _text_gts, logits,
    #         suffix='val'
    #     )

    #     self.validation_step_outputs.append([
    #         self.text_dec(logits.argmax(-1).detach().cpu().numpy()),
    #         text_gts
    #     ])

    # def on_validation_epoch_end(self):

    #     preds, gts = [], []
    #     for pred, gt in self.validation_step_outputs:
    #         preds.extend(pred)
    #         gts.extend(gt)

    #     df = pd.DataFrame({
    #         'gt': gts,
    #         'pred': preds,
    #     })

    #     acc = self._pd_get_text_acc(df)
    #     anls = self._pd_cal_anls_score(df)

    #     print('\n')
    #     print(tabulate(
    #         {
    #             'Dataset': ['MRZBenchmarkDataset'],
    #             'ACC': [acc],
    #             'ANLS': [anls]
    #         },
    #         headers=['Dataset', 'ACC', 'ANLS'],
    #         tablefmt='psql'
    #     ))
    #     print('\n')

    #     self.log('val_anls', torch.from_numpy(anls), sync_dist=True)
    #     self.validation_step_outputs.clear()

    # def _get_text_acc(self, preds, gts):
    #     return sum([i.lower() == j.lower() for i, j in zip(preds, gts)]) / len(preds)

    # def _cal_anls_score(self, preds, gts):
    #     return self.anls_text(preds, gts).cpu().numpy()

    # def _pd_get_text_acc(self, df):
    #     return self._get_text_acc(df['pred'].tolist(), df['gt'].tolist())

    # def _pd_cal_anls_score(self, df):
    #     return self._cal_anls_score(df['pred'].tolist(), df['gt'].tolist())

    # def _text_decode(self, text_preds, text_gts):
        text_preds = text_preds.argmax(-1).detach().cpu().numpy()
        text_preds = self.text_dec(text_preds)
        text_gts = text_gts.detach().cpu().numpy()
        text_gts = self.text_dec(text_gts)
        return text_preds, text_gts

    # @property
    # def preview_dir(self):
    #     img_path = Path(self.cfg.root_dir) / "preview" / \
    #         f'epoch_{self.current_epoch}'
    #     if not img_path.exists():
    #         img_path.mkdir(parents=True)
    #     return img_path

    # def preview(
    #     self, batch_idx, imgs,
    #     text_gts, text_logits,
    #     suffix='train'
    # ):

    #     # setup preview dir
    #     preview_dir = self.preview_dir / f'{suffix}_batch_{batch_idx}'
    #     if not preview_dir.exists():
    #         preview_dir.mkdir(parents=True)

    #     imgs = imgs.detach().cpu().numpy()
    #     text_preds, text_gts = self._text_decode(text_logits, text_gts)

    #     for idx, (img, text_gt, text_pred) in \
    #             enumerate(zip(imgs, text_gts, text_preds)):

    #         # Image
    #         img = np.uint8(np.transpose(img, (1, 2, 0)) * 255).copy()

    #         infos = {
    #             'Length': len(text_gt),
    #             'Text GT': text_gt,
    #             'Text Pred': text_pred,
    #             'Text ANLS': self.anls_text(text_pred, text_gt).tolist(),
    #         }

    #         if text_gt == text_pred:
    #             continue

    #         img_output_name = str(preview_dir / f'{idx}.jpg')
    #         imwrite(img, img_output_name)
    #         dump_json(infos, preview_dir / f'{idx}.json')
