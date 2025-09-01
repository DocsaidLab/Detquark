from __future__ import annotations

from typing import List, Tuple

import timm
import torch
import torch.nn as nn

try:
    from timm.models.registry import register_model
except ModuleNotFoundError:  # pragma: no cover
    def register_model(fn):  # type: ignore
        """Fallback decorator for environments without timm."""
        return fn

__all__ = [
    "TimmBackbone",
    "timm_backbone_base",
]


class TimmBackbone(nn.Module):
    """
    Wrapper for timm backbones that produces pyramid feature maps compatible with YOLO.

    This class aligns semantically with `YoloV11Backbone`:
    - Feature map shapes (spatial dimensions and channels) are derived from the timm
        model's `feature_info`.
    - Weights are initialized by timm; override `_init_weights()` to customize.

    Args:
        name (str): Name of the timm model (must support `features_only=True`).
        out_indices (Tuple[int, ...], optional): Indices of feature maps to return.
            Defaults to (0, 1, 2, 3, 4).
        pretrained (bool, optional): If True, load official timm pretrained weights.
            Defaults to False.
        features_only (bool, optional): Must be True. If False, it will be overridden.
            Defaults to True.
        **kwargs: Additional arguments passed to `timm.create_model`.
    """

    def __init__(
        self,
        name: str,
        *,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4),
        pretrained: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Create timm backbone with feature map outputs
        self.backbone: nn.Module = timm.create_model(
            name,
            features_only=True,
            pretrained=pretrained,
            **kwargs,
        )

        self.out_indices: Tuple[int, ...] = out_indices

        # Build feature metadata to match YoloV11Backbone interface
        self.feature_info: List[dict] = []
        for stage_idx, info in enumerate(self.backbone.feature_info, start=1):
            self.feature_info.append({
                "num_chs": info["num_chs"],
                "reduction": info["reduction"],
                "module": f"stage{stage_idx}",
                "stage": stage_idx,
            })

        # Expose channels per feature map stage
        self.channels: List[int] = [
            info["num_chs"]
            for info in self.feature_info
        ]

        # ------ compatibility shims --------------------------------------
        # Some neck or head needs these attributes to be set for compatibility
        # with other components that expect a backbone interface.
        default_compatibility = {
            "depth_mul": 1.0,
            "width_mul": 1.0,
            "max_channels": max(self.channels),
            "variant": "n",
        }
        default_compatibility.update(**kwargs)

        self.depth_mul = default_compatibility["depth_mul"]
        self.width_mul = default_compatibility["width_mul"]
        self.max_channels = default_compatibility["max_channels"]
        self.variant = default_compatibility["variant"]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # noqa: D401
        """
        Extract feature maps at specified indices.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: Output feature maps.
        """
        features = self.backbone(x)
        return [features[i] for i in self.out_indices]
