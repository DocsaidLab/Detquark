from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from ...blocks import (CSPKernelMixFastBottleneckBlock,
                       SpatialPyramidPoolingFastBlock)
from ...layers import ConvBNActivation, make_divisible

try:
    from timm.models.registry import register_model
except ModuleNotFoundError:
    def register_model(fn):
        """Fallback decorator for environments without timm."""
        return fn


__all__ = [
    "YoloV11Backbone",
    "yolov11_backbone_n",
    "yolov11_backbone_s",
    "yolov11_backbone_m",
    "yolov11_backbone_l",
    "yolov11_backbone_x",
]


class YoloV11Backbone(nn.Module):
    """YOLO-v11 backbone producing pyramid feature maps P1-P5 with strides 2-32.

    This implementation follows timm's `features_only` convention, returning
    feature maps in increasing order of stride.

    Args:
        variant (str): Scale variant, one of ['n', 's', 'm', 'l', 'x'].
        in_chans (int): Number of input channels. Default is 3.
        out_indices (Tuple[int, ...]): Which feature map indices to return.
            Defaults to (0, 1, 2, 3, 4), corresponding to P1-P5.

    Raises:
        ValueError: If `variant` is not supported.
    """

    # Configuration: (depth_mul, width_mul, max_channels)
    _variant_cfg = {
        "n": (0.50, 0.25, 1024),
        "s": (0.50, 0.50, 1024),
        "m": (0.50, 1.00,  512),
        "l": (1.00, 1.00,  512),
        "x": (1.00, 1.50,  512),
    }

    def __init__(
        self,
        variant: str = "n",
        in_chans: int = 3,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4),
    ) -> None:
        super().__init__()
        if variant not in self._variant_cfg:
            raise ValueError(f"Unsupported variant '{variant}'")

        depth_mul, width_mul, c_max = self._variant_cfg[variant]

        self.depth_mul = depth_mul
        self.width_mul = width_mul
        self.max_channels = c_max
        self.variant = variant

        def c(base: int) -> int:
            """Compute channel count: width multiplier then divisible by 8."""
            return make_divisible(min(base, c_max) * width_mul, 8)

        def depth(reps: int) -> int:
            """Compute block repeat count: at least 1."""
            return max(1, round(reps * depth_mul))

        # Define channels for each stage
        c1, c2, c3, c4, c5 = map(c, (64, 128, 256, 512, 1024))

        # Compute repeats for different stages
        rep_small = depth(2)   # for stage2 and part of stage5
        rep_mid = depth(2)     # for stage3
        rep_large = depth(2)   # for stage4

        # ---------- build network layers ----------
        # P1: stem convolution, downsample by 2
        self.stem = ConvBNActivation(in_chans, c1, kernel_size=3, stride=2)

        # P2: downsample to stride 4 + CSP block
        self.down2 = ConvBNActivation(c1, c2, kernel_size=3, stride=2)
        self.stage2 = CSPKernelMixFastBottleneckBlock(
            c2, c3,
            num_blocks=rep_small,
            use_c3k=True if variant in "mlx" else False,  # n variant uses C3k
            expansion=0.25,
        )

        # P3: downsample to stride 8 + CSP block
        self.down3 = ConvBNActivation(c3, c3, kernel_size=3, stride=2)
        self.stage3 = CSPKernelMixFastBottleneckBlock(
            c3, c4,
            num_blocks=rep_mid,
            use_c3k=True if variant in "mlx" else False,  # n variant uses C3k
            expansion=0.25,
        )

        # P4: downsample to stride 16 + CSP block
        self.down4 = ConvBNActivation(c4, c4, kernel_size=3, stride=2)
        self.stage4 = CSPKernelMixFastBottleneckBlock(
            c4, c4,
            num_blocks=rep_large,
            use_c3k=True,
            expansion=0.5,
        )

        # P5: downsample to stride 32 + CSP + SPPF + PSA stack
        self.down5 = ConvBNActivation(c4, c5, kernel_size=3, stride=2)
        self.stage5 = nn.Sequential(
            CSPKernelMixFastBottleneckBlock(
                c5, c5,
                num_blocks=rep_small,
                use_c3k=True,
                expansion=0.5,
            ),
            SpatialPyramidPoolingFastBlock(c5, c5, kernel_size=5)
        )

        # Metadata for timm compatibility
        self.out_indices = out_indices
        self.feature_info = [
            {"num_chs": c1, "reduction": 2,  "module": "stem",   "stage": 1},
            {"num_chs": c2, "reduction": 4,  "module": "stage2", "stage": 2},
            {"num_chs": c3, "reduction": 8,  "module": "stage3", "stage": 3},
            {"num_chs": c4, "reduction": 16, "module": "stage4", "stage": 4},
            {"num_chs": c5, "reduction": 32, "module": "stage5", "stage": 5},
        ]

        self._init_weights()

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract pyramid features P1-P5 from input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_chans, H, W).

        Returns:
            List[torch.Tensor]: Selected feature maps as per `out_indices`.
        """
        p1 = self.stem(x)
        p2 = self.down2(p1)
        p3 = self.down3(self.stage2(p2))
        p4 = self.down4(self.stage3(p3))
        p5 = self.stage5(self.down5(self.stage4(p4)))
        feats = (p1, p2, p3, p4, p5)

        return [feats[i] for i in self.out_indices]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Alias for `forward_features` to match nn.Module interface."""
        return self.forward_features(x)

    def _init_weights(self) -> None:
        """Initialize weights using best practices for SiLU-based convolutional neural networks.

        - Conv2d (point-wise):
            Kaiming normal initialization with fan_out mode and SiLU-equivalent gain.
        - Conv2d (depth-wise):
            Kaiming normal initialization with fan_in mode.
        - BatchNorm layers:
            weight (gamma) set to 1, bias (beta) set to 0, running_mean zeroed, and running_var set to 1.
        - Linear layers:
            truncated normal initialization with std=0.02.
        - Other modules: PyTorch default initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Determine if this is a depth-wise convolution
                is_depthwise = (
                    module.groups == module.in_channels == module.out_channels
                )
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in" if is_depthwise else "fan_out",
                    nonlinearity="linear",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                # Initialize BatchNorm: gamma=1, beta=0, reset running stats
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                module.running_mean.zero_()
                module.running_var.fill_(1.0)

            elif isinstance(module, nn.Linear):
                # Initialize Linear layers with truncated normal distribution
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm layers: gamma=1, beta=0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def _create_backbone(variant: str, pretrained: bool = False, **kwargs) -> YoloV11Backbone:
    """Helper to instantiate a YoloV11Backbone with optional pretrained flag.

    Args:
        variant (str): Model variant key.
        pretrained (bool): If True, notifies that pretrained weights are unavailable.
        **kwargs: Additional keyword arguments passed to YoloV11Backbone.

    Returns:
        YoloV11Backbone: Instantiated backbone model.
    """
    if pretrained:
        print("Pretrained weights are not available for YoloV11Backbone.")
    out_indices = kwargs.pop("out_indices", (0, 1, 2, 3, 4))
    if kwargs:
        print("Additional parameters are not implemented for YoloV11Backbone.")
    return YoloV11Backbone(variant=variant, out_indices=out_indices)


@register_model
def yolov11_backbone_n(pretrained: bool = False, **kwargs) -> YoloV11Backbone:
    """YoloV11 backbone variant 'n'."""
    return _create_backbone("n", pretrained, **kwargs)


@register_model
def yolov11_backbone_s(pretrained: bool = False, **kwargs) -> YoloV11Backbone:
    """YoloV11 backbone variant 's'."""
    return _create_backbone("s", pretrained, **kwargs)


@register_model
def yolov11_backbone_m(pretrained: bool = False, **kwargs) -> YoloV11Backbone:
    """YoloV11 backbone variant 'm'."""
    return _create_backbone("m", pretrained, **kwargs)


@register_model
def yolov11_backbone_l(pretrained: bool = False, **kwargs) -> YoloV11Backbone:
    """YoloV11 backbone variant 'l'."""
    return _create_backbone("l", pretrained, **kwargs)


@register_model
def yolov11_backbone_x(pretrained: bool = False, **kwargs) -> YoloV11Backbone:
    """YoloV11 backbone variant 'x'."""
    return _create_backbone("x", pretrained, **kwargs)
