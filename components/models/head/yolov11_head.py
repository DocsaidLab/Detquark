import math
from typing import List, Tuple

import torch
import torch.nn as nn

from ...blocks import DFLIntegral
from ...layers import ConvBNActivation, DWConv


class YOLOv11Head(nn.Module):
    """Anchor-free detection head for YOLOv11.

    This head produces bounding box distributions and class logits for each
    scale level in a feature pyramid. Bounding box distributions are converted
    to continuous offsets using Distribution-Focal Loss (DFL) integration.

    Attributes:
        num_classes (int): Number of target classes to predict.
        num_levels (int): Number of feature map levels.
        reg_max (int): Number of discrete bins for the distribution.
        strides (List[int]): Stride values for each feature level.
        box_convs (nn.ModuleList): Sequential layers predicting box distributions.
        cls_convs (nn.ModuleList): Sequential layers predicting class logits.
        dfl (nn.Module): Module for converting discrete distributions to offsets.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int = 80,
        reg_max: int = 16,
        strides: Tuple[int, ...] = (8, 16, 32),
        **kwargs
    ) -> None:
        """Initializes the YOLOv11 detection head.

        Args:
            in_channels_list (List[int]): Number of channels for each input feature.
            num_classes (int, optional): Number of object classes. Defaults to 80.
            reg_max (int, optional): Number of regression bins. Defaults to 16.
            strides (Tuple[int, ...], optional): Stride for each feature. Defaults to (8,16,32).
            **kwargs: Additional keyword arguments (unused).

        Raises:
            ValueError: If lengths of in_channels_list and strides do not match.
        """
        super().__init__()
        if len(in_channels_list) != len(strides):
            raise ValueError(
                "Length of in_channels_list must match length of strides"
            )

        # Core parameters
        self.num_classes: int = num_classes
        self.num_levels: int = len(in_channels_list)
        self.reg_max: int = reg_max
        self.strides: List[int] = list(strides)

        # Determine hidden channel sizes for box and class branches
        hidden_box_ch = max(16, in_channels_list[0] // 4, 4 * reg_max)
        hidden_cls_ch = max(in_channels_list[0], min(num_classes, 100))

        # Convolutions predicting bounding box distributions
        self.box_convs = nn.ModuleList([
            nn.Sequential(
                ConvBNActivation(c_in, hidden_box_ch, kernel_size=3),
                ConvBNActivation(hidden_box_ch, hidden_box_ch, kernel_size=3),
                nn.Conv2d(hidden_box_ch, 4 * reg_max, kernel_size=1),
            ) for c_in in in_channels_list
        ])

        # Convolutions predicting class logits
        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                DWConv(c_in, c_in, kernel_size=3),
                ConvBNActivation(c_in, hidden_cls_ch, kernel_size=1),
                DWConv(hidden_cls_ch, hidden_cls_ch, kernel_size=3),
                ConvBNActivation(hidden_cls_ch, hidden_cls_ch, kernel_size=1),
                nn.Conv2d(hidden_cls_ch, num_classes, kernel_size=1),
            ) for c_in in in_channels_list
        ])

        # Distribution-Focal Loss integral operator or identity if disabled
        self.dfl: nn.Module = (
            DFLIntegral(reg_max) if reg_max > 1 else nn.Identity()
        )

        self._init_bias()

    def _init_bias(self) -> None:
        for s, box_seq, cls_seq in zip(self.strides, self.box_convs, self.cls_convs):
            box_seq[-1].bias.data[:] = 1.0
            cls_seq[-1].bias.data[: self.num_classes] = math.log(
                5 / self.num_classes / (640 / s) ** 2)

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Forward pass for all feature levels.

        Args:
            features (List[torch.Tensor]): List of feature maps from FPN/PAN,
                each with shape (batch_size, channels, height, width).

        Returns:
            List[torch.Tensor]: List of output tensors for each level. Each output
                has shape (batch_size, 4*reg_max + num_classes, height, width),
                concatenating box and class predictions.

        Raises:
            ValueError: If number of feature maps does not match expected levels.
        """
        if len(features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature maps, got {len(features)}"
            )

        outputs: List[torch.Tensor] = []
        for level, feat in enumerate(features):
            # Predict raw distribution logits for bounding boxes
            box_logits = self.box_convs[level](feat)
            # Predict raw class logits
            cls_logits = self.cls_convs[level](feat)
            # Concatenate box and class predictions
            outputs.append(torch.cat((box_logits, cls_logits), dim=1))

        return outputs
