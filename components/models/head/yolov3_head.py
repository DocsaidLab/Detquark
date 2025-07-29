from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn


class YOLOv3Head(nn.Module):
    """YOLOv3 detection head with multi-scale anchor-based predictions.

    Attributes:
        num_classes (int): Number of object classes.
        n_scales (int): Number of prediction scales.
        strides (List[int]): Downsampling strides for each scale.
        anchors (Tensor): Global anchor sizes in pixels, shape (A, 2).
        pred_layers (nn.ModuleList): Per-scale prediction convolutional layers.
        anchor_cells (List[Tensor]): Anchor sizes normalized by stride per scale.
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
        self.anchor_cells: List[torch.Tensor] = []

        for i, (c_in, mask, stride) in enumerate(zip(in_channels_list, anchor_masks, strides)):
            mask = list(mask)
            num_anchors = len(mask)
            anchor_cell = self.anchors[mask] / stride
            self.register_buffer(f"anchor_cells_s{i}", anchor_cell)
            self.anchor_cells.append(anchor_cell)

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
