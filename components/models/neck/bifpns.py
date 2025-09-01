from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["BiFPN", "BiFPNs"]


def build_activation(act: Optional[Union[str, dict, nn.Module]]) -> Optional[nn.Module]:
    if act is None:
        return None
    if isinstance(act, nn.Module):
        return act
    if isinstance(act, str):
        name, kwargs = act, {}
    elif isinstance(act, dict):
        name = act.get("name", "ReLU")
        kwargs = {k: v for k, v in act.items() if k != "name"}
    else:
        raise TypeError(f"Unsupported activation spec: {type(act)}")

    name_l = str(name).lower()
    if name_l in ("relu",):
        return nn.ReLU(**kwargs)
    if name_l in ("relu6",):
        return nn.ReLU6(**kwargs)
    if name_l in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(**kwargs)
    if name_l in ("silu", "swish"):
        return nn.SiLU(**kwargs)
    if name_l in ("gelu",):
        return nn.GELU(**kwargs)
    if name_l in ("hardswish", "hswish"):
        return nn.Hardswish()
    if name_l in ("mish",):
        return nn.Mish(**kwargs)
    if name_l in ("elu",):
        return nn.ELU(**kwargs)
    # fallback
    return nn.ReLU(**kwargs)


def build_norm2d(norm: Optional[Union[dict, nn.Module]], num_features: Optional[int] = None) -> Optional[nn.Module]:
    if norm is None:
        return None
    if isinstance(norm, nn.Module):
        return norm
    if not isinstance(norm, dict):
        raise TypeError(f"Unsupported norm spec: {type(norm)}")

    name = norm.get("name", "BatchNorm2d")
    name_l = str(name).lower()
    # Common kwargs
    eps = norm.get("eps", 1e-5)
    momentum = norm.get("momentum", 0.1)
    affine = norm.get("affine", True)

    if name_l in ("batchnorm2d", "bn2d", "bn"):
        nf = norm.get("num_features", num_features)
        if nf is None:
            raise ValueError("BatchNorm2d requires `num_features`.")
        track = norm.get("track_running_stats", True)
        return nn.BatchNorm2d(nf, eps=eps, momentum=momentum, affine=affine, track_running_stats=track)

    if name_l in ("syncbatchnorm", "syncbn"):
        nf = norm.get("num_features", num_features)
        if nf is None:
            raise ValueError("SyncBatchNorm requires `num_features`.")
        # torch SyncBN ignores track_running_stats arg
        return nn.SyncBatchNorm(nf, eps=eps, momentum=momentum, affine=affine)

    if name_l in ("instancenorm2d", "in2d"):
        nf = norm.get("num_features", num_features)
        if nf is None:
            raise ValueError("InstanceNorm2d requires `num_features`.")
        track = norm.get("track_running_stats", False)
        return nn.InstanceNorm2d(nf, eps=eps, momentum=momentum, affine=affine, track_running_stats=track)

    if name_l in ("groupnorm", "gn"):
        num_groups = norm.get("num_groups", 32)
        # GroupNorm 使用 num_channels 參數；若未提供則回退到 num_features
        nc = norm.get("num_channels", norm.get("num_features", num_features))
        if nc is None:
            raise ValueError("GroupNorm requires `num_channels`.")
        return nn.GroupNorm(num_groups=num_groups, num_channels=nc, eps=eps, affine=affine)

    if name_l in ("layernorm", "ln"):
        # 注意：CNN 上使用 LayerNorm 需配合 channels_last；此處僅提供構造
        normalized_shape = norm.get(
            "normalized_shape", norm.get("num_features", num_features))
        if normalized_shape is None:
            raise ValueError(
                "LayerNorm requires `normalized_shape` or `num_features`.")
        elementwise_affine = norm.get("elementwise_affine", True)
        return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    raise ValueError(f"Unsupported norm name: {name}")


def clone_norm_for_channels(norm: Optional[Union[dict, nn.Module]], num_features: int) -> Optional[Union[dict, nn.Module]]:
    """Clone/adjust a norm spec to match a specific channel count (used by extra layers)."""
    if norm is None:
        return None
    if isinstance(norm, dict):
        cfg = dict(norm)
        cfg["num_features"] = num_features
        # For GroupNorm, prefer num_channels key
        if str(cfg.get("name", "")).lower() in ("groupnorm", "gn"):
            cfg["num_channels"] = num_features
        return cfg

    # If module instance, attempt best-effort reconstruction
    if isinstance(norm, nn.BatchNorm2d):
        return nn.BatchNorm2d(num_features, eps=norm.eps, momentum=norm.momentum, affine=norm.affine, track_running_stats=norm.track_running_stats)
    if isinstance(norm, nn.SyncBatchNorm):
        return nn.SyncBatchNorm(num_features, eps=norm.eps, momentum=norm.momentum, affine=norm.affine)
    if isinstance(norm, nn.InstanceNorm2d):
        return nn.InstanceNorm2d(num_features, eps=norm.eps, momentum=norm.momentum, affine=norm.affine, track_running_stats=norm.track_running_stats)
    if isinstance(norm, nn.GroupNorm):
        return nn.GroupNorm(norm.num_groups, num_features, eps=norm.eps, affine=norm.affine)
    return None


def _init_conv_weight(w: torch.Tensor, init_type: str = "kaiming") -> None:
    it = str(init_type).lower()
    if it in ("kaiming", "kaiming_normal", "he"):
        nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
    elif it in ("kaiming_uniform", "he_uniform"):
        nn.init.kaiming_uniform_(w, mode="fan_out", nonlinearity="relu")
    elif it in ("xavier", "xavier_uniform"):
        nn.init.xavier_uniform_(w)
    elif it in ("xavier_normal",):
        nn.init.xavier_normal_(w)
    elif it in ("normal",):
        nn.init.normal_(w, mean=0.0, std=0.01)
    else:
        nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")


def _init_norm(m: nn.Module) -> None:
    if hasattr(m, "weight") and m.weight is not None:
        nn.init.ones_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.zeros_(m.bias)


class WeightedSum(nn.Module):
    def __init__(
        self,
        input_size: int,
        act: Optional[Union[dict, nn.Module]] = None,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.weights = nn.Parameter(torch.ones(
            input_size, dtype=torch.float32), requires_grad=requires_grad)
        self.weights_relu = nn.ReLU()
        self.relu = build_activation(act) if act is not None else nn.Identity()
        self.epsilon = 1e-4

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if len(x) != self.input_size:
            raise ValueError("Invalid input size not equal to weight size.")
        weights = self.weights_relu(self.weights)
        weights = weights / \
            (torch.sum(weights, dim=0, keepdim=True) + self.epsilon)
        # x: list of [N,C,H,W], stack-> [K,N,C,H,W], einsum 'i,i...->...' 做加權
        weighted_x = torch.einsum(
            "i,i...->...", weights, torch.stack(x, dim=0))
        return self.relu(weighted_x)


class SeparableConv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        bias: bool = False,
        inner_norm: Optional[Union[dict, nn.Module]] = None,
        inner_act: Optional[Union[dict, nn.Module]] = None,
        norm: Optional[Union[dict, nn.Module]] = None,
        act: Optional[Union[dict, nn.Module]] = None,
        init_type: str = "normal",
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        bias = False if norm is not None else bias

        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.inner_norm = build_norm2d(
            inner_norm, num_features=in_channels) if inner_norm is not None else None
        self.inner_act = build_activation(
            inner_act) if inner_act is not None else None

        self.pw_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.norm = build_norm2d(
            norm, num_features=out_channels) if norm is not None else None
        self.act = build_activation(act) if act is not None else None

        self.initialize_weights_(init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        if self.inner_norm is not None:
            x = self.inner_norm(x)
        if self.inner_act is not None:
            x = self.inner_act(x)
        x = self.pw_conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def initialize_weights_(self, init_type: str = "kaiming") -> None:
        _init_conv_weight(self.dw_conv.weight, init_type)
        _init_conv_weight(self.pw_conv.weight, init_type)
        if self.pw_conv.bias is not None:
            nn.init.zeros_(self.pw_conv.bias)
        if self.inner_norm is not None:
            _init_norm(self.inner_norm)
        if self.norm is not None:
            _init_norm(self.norm)


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: Union[float, int],
        out_channels: Union[float, int],
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        norm: Optional[Union[dict, nn.Module]] = None,
        act: Optional[Union[dict, nn.Module]] = None,
        init_type: str = "normal",
    ):
        super().__init__()
        bias = False if norm is not None else bias

        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.norm = build_norm2d(norm, num_features=int(
            out_channels)) if norm is not None else None
        self.act = build_activation(act) if act is not None else None

        self.initialize_weights_(init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def initialize_weights_(self, init_type: str = "kaiming") -> None:
        _init_conv_weight(self.conv.weight, init_type)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if self.norm is not None:
            _init_norm(self.norm)


class BiFPN(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_layers: int = 0,
        out_indices: Optional[List[int]] = None,
        norm: Optional[Union[dict, nn.Module]] = None,
        act: Optional[Union[dict, nn.Module]] = None,
        upsample_mode: str = "bilinear",
        use_conv: bool = False,
        attention: bool = True,
    ) -> None:
        super().__init__()

        if extra_layers < 0:
            raise ValueError("extra_layers < 0, which is invalid.")

        self.attention = attention
        self.upsample_mode = upsample_mode
        self.in_channels_list = in_channels_list

        num_in_features = len(in_channels_list)
        num_out_features = num_in_features + extra_layers

        Conv = Conv2dBlock if use_conv else SeparableConv2dBlock

        # ───────── Lateral 1x1（對齊到 out_channels） ─────────
        conv1x1s: List[nn.Module] = []
        for i in range(num_out_features):
            in_ch = in_channels_list[i] if i < num_in_features else in_channels_list[-1]
            if in_ch != out_channels:
                conv1x1s.append(
                    Conv(
                        in_channels=in_ch,
                        out_channels=out_channels,
                        kernel=1,
                        stride=1,
                        padding=0,
                        norm=norm,
                    )
                )
            else:
                conv1x1s.append(nn.Identity())
        self.conv1x1s = nn.ModuleList(conv1x1s)

        # ───────── Top-down / Bottom-up 的 3x3 conv（平滑）─────────
        self.conv_up_3x3s = nn.ModuleList(
            [
                Conv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel=3,
                    stride=1,
                    padding=1,
                    norm=norm,
                    act=act,
                )
                for _ in range(num_out_features - 1)
            ]
        )

        self.conv_down_3x3s = nn.ModuleList(
            [
                Conv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel=3,
                    stride=1,
                    padding=1,
                    norm=norm,
                    act=act,
                )
                for _ in range(num_out_features - 1)
            ]
        )

        # ───────── 最高層以後的擴展層（如 P6/P7）─────────
        if extra_layers > 0:
            extra_norm = clone_norm_for_channels(norm, in_channels_list[-1])
            self.extra_conv_downs = nn.ModuleList(
                [
                    Conv(
                        in_channels=in_channels_list[-1],
                        out_channels=in_channels_list[-1],
                        kernel=3,
                        stride=2,
                        padding=1,
                        norm=extra_norm,
                        act=act,
                    )
                    for _ in range(extra_layers)
                ]
            )

        # ───────── 尺度運算 ─────────
        self.upsamples = nn.ModuleList(
            [
                nn.Upsample(
                    scale_factor=2,
                    mode=upsample_mode,
                    align_corners=False if upsample_mode != "nearest" else None,
                )
                for _ in range(num_out_features - 1)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                for _ in range(num_out_features - 1)
            ]
        )

        # ───────── 加權和（BiFPN Attention）─────────
        act_relu = {"name": "ReLU", "inplace": True}
        self.weighted_sum_2_input = nn.ModuleList(
            [
                WeightedSum(input_size=2, act=act_relu,
                            requires_grad=attention)
                for _ in range(num_out_features)
            ]
        )
        self.weighted_sum_3_input = nn.ModuleList(
            [
                WeightedSum(input_size=3, act=act_relu,
                            requires_grad=attention)
                for _ in range(num_out_features - 2)
            ]
        )

        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.out_indices = out_indices

        self.initialize_weights_()

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(xs) != self.num_in_features:
            raise ValueError(
                "The length of given xs is not equal to the length of in_channels_list.")

        # 需要時擴張額外層（最高層一路 stride=2）
        if hasattr(self, "extra_conv_downs"):
            extras: List[torch.Tensor] = []
            x = xs[-1]
            for conv in self.extra_conv_downs:
                x = conv(x)
                extras.append(x)
            xs = xs + extras  # 變成 num_out_features

        # 固定寬度
        out_fixed = [self.conv1x1s[i](xs[i]) for i in range(len(xs))]

        # ───────── Top-down ─────────
        outs_top_down: List[torch.Tensor] = []
        hidden: torch.Tensor = torch.empty(0)  # 將於第一輪覆蓋
        for i in range(len(out_fixed) - 1, -1, -1):
            out = out_fixed[i]
            if i != len(xs) - 1:
                hidden = self.weighted_sum_2_input[i](
                    [out, self.upsamples[i](hidden)])
                out = self.conv_up_3x3s[i](hidden)
            hidden = out
            outs_top_down.append(out)
        outs_top_down = outs_top_down[::-1]  # 低 -> 高

        # ───────── Bottom-up ─────────
        outs_down_top: List[torch.Tensor] = []
        hidden = torch.empty(0)
        for i in range(len(outs_top_down)):
            out = outs_top_down[i]
            residual = out_fixed[i]

            if i != 0 and i != len(outs_top_down) - 1:
                # 中間層：3 輸入加權
                hidden = self.weighted_sum_3_input[i - 1](
                    [out, self.downsamples[i - 1](hidden), residual]
                )
                out = self.conv_down_3x3s[i - 1](hidden)
            elif i == len(outs_top_down) - 1:
                # 最高層：2 輸入加權（索引使用 i 與層對齊）
                hidden = self.weighted_sum_2_input[i](
                    [self.downsamples[i - 1](hidden), residual]
                )
                out = self.conv_down_3x3s[i - 1](hidden)

            hidden = out
            outs_down_top.append(out)

        if self.out_indices is not None:
            outs_down_top = [outs_down_top[i] for i in self.out_indices]

        return outs_down_top

    def initialize_weights_(self) -> None:
        # 若子模組實作 reset_parameters 則呼叫（容錯）
        for m in self.modules():
            if hasattr(m, "reset_parameters") and callable(getattr(m, "reset_parameters")):
                try:
                    m.reset_parameters()
                except Exception:
                    pass


class BiFPNs(nn.Module):

    def __init__(
        self,
        in_channels_list: List[int],
        in_indices: List[int],
        n_blocks: int = 1,
        repeat: int = 1,  # 保留以對齊（未使用）
        depth_mul: float = 1.0,
        out_indices: Optional[List[int]] = None,
        *,
        out_channels: int,
        extra_layers: int = 0,
        upsample_mode: str = "nearest",
        attention: bool = True,
        use_conv: bool = False,
        variant: str = "n",  # 保留參數以對齊
        norm: Optional[Union[dict, nn.Module]] = None,
        act: Optional[Union[dict, nn.Module]] = None,
        **kwargs: object,
    ) -> None:
        super().__init__()

        if n_blocks < 1:
            raise ValueError("`n_blocks` must be >= 1.")
        if sorted(in_indices) != in_indices:
            raise ValueError(
                "`in_indices` must be sorted from low to high resolution.")

        self.in_indices = in_indices
        self.sel_channels = [in_channels_list[i] for i in in_indices]

        num_out_features = len(self.sel_channels) + extra_layers
        self._fused_channels = out_channels

        # 預設輸出：全部（含 extra_layers）
        self.out_indices = out_indices or list(range(num_out_features))

        # 堆疊層數（受 depth_mul 影響）
        stacked = max(1, round(n_blocks * depth_mul))

        # 建立 BiFPN 堆疊
        blocks: List[BiFPN] = []
        for i in range(stacked):
            blocks.append(
                BiFPN(
                    in_channels_list=self.sel_channels if i == 0 else [
                        out_channels] * num_out_features,
                    out_channels=out_channels,
                    extra_layers=extra_layers if i == 0 else 0,
                    out_indices=None,  # 統一在最外層切
                    norm=norm if isinstance(norm, (dict, nn.Module)) else {
                        "name": "BatchNorm2d", "num_features": out_channels, "momentum": 0.003, "eps": 1e-4
                    },
                    act=act if isinstance(act, (dict, nn.Module)) else {
                        "name": "ReLU", "inplace": False},
                    upsample_mode=upsample_mode,
                    use_conv=use_conv,
                    attention=attention,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        xs = [feats[i] for i in self.in_indices]
        for bifpn in self.blocks:
            xs = bifpn(xs)
        return [xs[i] for i in self.out_indices]

    @property
    def out_channels(self) -> List[int]:
        return [self._fused_channels for _ in self.out_indices]
