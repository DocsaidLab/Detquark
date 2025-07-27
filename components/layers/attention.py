import torch
import torch.nn as nn

from .common import ConvBNActivation

__all__ = [
    "ConvAttention",
    "PositionSensitiveAttention",
]


class ConvAttention(nn.Module):
    """Convolution-augmented multi-head self-attention (MHSA).

    Pipeline
    --------
    1. **Q K V projection** - 1 x 1 conv → split into *Q*, *K*, *V*
    2. **Attention** - scaled dot-product per head
    3. **Depth-wise positional encoding** - 3 x 3 depth-wise conv
    4. **Output projection** - 1 x 1 conv

    Args:
        dim (int):        Input channel dimension *C*.
        num_heads (int):  Number of attention heads *H*.
        attn_ratio (float, optional): Key/Query dimension ratio (``key_dim =
            attn_ratio · (C / H)``). Defaults to ``0.5``.
        activation (bool | nn.Module, optional): Activation for *proj* & *pe*
            layers (``qkv`` has **no** activation).

            * ``True`` → new SiLU instance (default)
            * ``False`` → identity
            * ``nn.Module`` → user-supplied module (deep-copied)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_ratio: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5

        nh_kd = self.key_dim * num_heads
        proj_channels = dim + nh_kd * 2  # q + k + v

        # --- Q K V projection (no BN / act) -------------------------------- #
        self.qkv = ConvBNActivation(
            dim,
            proj_channels,
            kernel_size=1,
            activation=False,  # <- ONNX-friendly
        )

        # --- Output projection --------------------------------------------- #
        self.proj = ConvBNActivation(
            dim,
            dim,
            kernel_size=1,
            activation=False,
        )

        # --- Depth-wise positional encoding -------------------------------- #
        self.pe = ConvBNActivation(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            groups=dim,
            activation=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MHSA with depth-wise positional encoding."""
        b, c, h, w = x.shape
        n = h * w  # sequence length

        # QKV projection and reshape to (B, H, Dq+K+V, N)
        qkv = self.qkv(x).reshape(b, self.num_heads, -1, n)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        # Scaled dot-product attention: (B, H, N, N)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # Weighted sum → (B, C, H, W)
        out = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w)

        # Positional encoding (depth-wise)
        out = out + self.pe(v.reshape(b, c, h, w))

        # Final projection
        return self.proj(out)


class PositionSensitiveAttention(nn.Module):
    """Position- Sensitive Attention (PSA) with optional residual shortcuts.

    Pipeline
    --------
    1. **MHSA** - convolutional self- attention
    2. **FFN**  - 1 x 1 → GELU → 1 x 1 (channel- wise MLP)
    3. **Shortcuts** - optional residual adds after each stage

    Args:
        channels (int):         Input/Output feature channels *C*.
        attn_ratio (float):     ``key_dim = attn_ratio · (C / heads)``.
        num_heads (int):        Number of attention heads.
        shortcut (bool):        Enable residual adds (default ``True``).
        activation (bool | nn.Module, optional): Activation for FFN and
            output projection in MHSA.

            * ``True`` → fresh SiLU (default)
            * ``False`` → identity
            * ``nn.Module`` → user- supplied module (deep- copied)
    """

    def __init__(
        self,
        channels: int,
        *,
        attn_ratio: float = 0.5,
        num_heads: int = 4,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        self.shortcut = shortcut

        # --- Attention --------------------------------------------------- #
        self.attn = ConvAttention(
            dim=channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio
        )

        # --- Feed- forward network (1×1 → act → 1×1) ---------------------- #
        self.ffn = nn.Sequential(
            ConvBNActivation(
                channels,
                channels * 2,
                kernel_size=1,
                activation=True,
            ),
            ConvBNActivation(
                channels * 2,
                channels,
                kernel_size=1,
                activation=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x → MHSA (+res) → FFN (+res)."""
        y = self.attn(x)
        x = x + y if self.shortcut else y

        y = self.ffn(x)
        x = x + y if self.shortcut else y
        return x
