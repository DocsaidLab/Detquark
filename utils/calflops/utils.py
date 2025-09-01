from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

DEFAULT_PRECISION: int = 2

__all__ = [
    "DEFAULT_PRECISION",
    "generate_transformer_input",
    "number_to_string",
    "macs_to_string",
    "flops_to_string",
    "bytes_to_string",
    "params_to_string",
    "get_module_flops",
    "get_module_macs",
    "convert_bytes",
    "is_package_available",
]


def generate_transformer_input(
    tokenizer: Any,
    input_shape: Optional[Tuple[int, int]] = None,
    device: Union[str, torch.device, None] = None,
    add_position_ids: bool = False,
) -> Dict[str, torch.Tensor]:
    """Generate batched, padded inputs for a Transformers model.

    This is useful for benchmarking or warm-up when you don't have real text.
    It creates a batch of empty strings and lets the tokenizer insert special
    tokens, then pads/truncates to `max_length`.

    Args:
        tokenizer: A HuggingFace-style tokenizer with `__call__` compatible
            signature (e.g., `AutoTokenizer`). Must support `padding`, `max_length`,
            `truncation`, and `return_tensors="pt"`.
        input_shape: Pair `(batch_size, seq_len)`. If `None`, defaults to `(1, 128)`.
        device: Target device for the returned tensors. If `None`, tensors remain
            on CPU.
        add_position_ids: If `True`, also include a `position_ids` tensor of shape
            `(batch, seq_len)` with values `[0..seq_len-1]`. Many models construct
            these internally and don't require it; set this only if your model
            explicitly needs it (e.g., some ChatGLM variants).

    Returns:
        A dict containing at least `input_ids` and `attention_mask`, and including
        `token_type_ids` and/or `position_ids` if applicable.

    Raises:
        ValueError: If `input_shape` is not a (batch, seq_len) pair or contains
            non-positive integers.
    """
    if input_shape is None:
        input_shape = (1, 128)

    if not (isinstance(input_shape, (tuple, list)) and len(input_shape) == 2):
        raise ValueError(
            "`input_shape` must be a tuple/list of length 2 (batch, seq_len).")

    batch_size, max_length = int(input_shape[0]), int(input_shape[1])
    if batch_size <= 0 or max_length <= 0:
        raise ValueError(
            "Both batch size and seq_len must be positive integers.")

    # Tokenize a batch of empty prompts; tokenizer will add special tokens.
    texts = [""] * batch_size
    encoded = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    inputs: Dict[str, torch.Tensor] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    # Not all tokenizers return token_type_ids.
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"]

    # Some tokenizers might supply `position_ids`; otherwise optionally build it.
    if "position_ids" in encoded:
        inputs["position_ids"] = encoded["position_ids"]
    elif add_position_ids:
        position_ids = torch.arange(
            max_length, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        inputs["position_ids"] = position_ids

    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


# --- Formatting helpers -------------------------------------------------------

_SI_UNITS = {
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "K": 1e3,
    "": 1.0,
    "m": 1e-3,
    "u": 1e-6,
}


def number_to_string(
    num: Union[int, float],
    units: Optional[str] = None,
    precision: int = DEFAULT_PRECISION
) -> str:
    """Format a number with SI-like unit prefixes.

    Supports tera (T), giga (G), mega (M), kilo (K), milli (m), micro (u).
    If `units` is None, a suitable unit is chosen automatically.

    Args:
        num: Number to format.
        units: Optional unit prefix among {'T','G','M','K','', 'm','u'}.
        precision: Decimal digits to keep.

    Returns:
        A human-friendly string such as '1.23 G', '456 m', or '789'.
    """
    sign = "-" if num < 0 else ""
    abs_num = float(abs(num))

    if units is None:
        if abs_num >= 1e12:
            units = "T"
        elif abs_num >= 1e9:
            units = "G"
        elif abs_num >= 1e6:
            units = "M"
        elif abs_num >= 1e3:
            units = "K"
        elif abs_num >= 1 or abs_num == 0:
            units = ""
        elif abs_num >= 1e-3:
            units = "m"
        else:
            units = "u"

    magnitude = _SI_UNITS.get(units, 1.0)
    value = abs_num / magnitude

    # Trim trailing zeros while respecting `precision`.
    s_val = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return f"{sign}{s_val} {units}".rstrip()


def macs_to_string(
    macs: Union[int, float],
    units: Optional[str] = None,
    precision: int = DEFAULT_PRECISION
) -> str:
    """Convert MACs (multiply-accumulate ops) to a string with unit suffix.

    Args:
        macs: MACs count.
        units: Desired unit prefix or `None` for auto.
        precision: Decimal digits to keep.

    Returns:
        MACs string, e.g. '12.3 GMACs'.
    """
    base = number_to_string(macs, units=units, precision=precision)
    return f"{base}MACs".replace("  ", " ")


def flops_to_string(
    flops: Union[int, float],
    units: Optional[str] = None,
    precision: int = DEFAULT_PRECISION
) -> str:
    """Convert FLOPs to a string with unit suffix.

    Args:
        flops: FLOPs count.
        units: Desired unit prefix or `None` for auto.
        precision: Decimal digits to keep.

    Returns:
        FLOPs string, e.g. '45.6 GFLOPs'.
    """
    base = number_to_string(flops, units=units, precision=precision)
    return f"{base}FLOPs".replace("  ", " ")


def bytes_to_string(
    b: Union[int, float],
    units: Optional[str] = None,
    precision: int = DEFAULT_PRECISION
) -> str:
    """Convert bytes to a string like '1.5 GB'.

    Args:
        b: Byte count.
        units: Desired unit prefix (T/G/M/K/'') or `None` for auto.
        precision: Decimal digits to keep.

    Returns:
        Human-friendly byte string.
    """
    base = number_to_string(b, units=units, precision=precision)
    return f"{base}B".replace("  ", " ")


def params_to_string(
    params_num: Union[int, float],
    units: Optional[str] = None,
    precision: int = DEFAULT_PRECISION
) -> str:
    """Convert parameter counts to a string using 'B' for billions.

    By convention, many papers label billions of parameters with 'B' instead of
    'G'. To support that, if `units=='B'` we internally map to 'G' and then
    switch the label back.

    Args:
        params_num: Number of parameters.
        units: Optional unit among {'T','G','M','K','', 'm','u','B'}. If 'B' is
            given, it will display billions as 'B' and trillions as 'TB' etc.
        precision: Decimal digits to keep.

    Returns:
        Parameter count string, e.g. '0.75 B' or '12.3 M'.
    """
    display_units = units
    internal_units = None
    if units is not None:
        internal_units = "G" if units == "B" else units

    s = number_to_string(params_num, units=internal_units, precision=precision)
    if display_units == "B":
        s = s.replace("G", "B")
    return s


# --- Model accounting helpers -------------------------------------------------

def _sparse_factor(module: nn.Module) -> float:
    """Compute non-zero ratio over trainable parameters for sparsity adjustment."""
    nz = 0
    total = 0
    for p in module.parameters(recurse=False):
        if p.requires_grad:
            total += p.numel()
            nz += int((p != 0).sum().item())
    return (nz / total) if total > 0 else 1.0


def get_module_flops(module: nn.Module, is_sparse: bool = False) -> float:
    """Recursively sum `__flops__` across a module hierarchy.

    Args:
        module: A `torch.nn.Module` whose submodules may define `__flops__`
            attributes (as produced by some profiling tools).
        is_sparse: If `True`, scale each module's self FLOPs by the ratio of
            non-zero to total trainable parameters in that module.

    Returns:
        Total FLOPs as a float.
    """
    self_flops = float(getattr(module, "__flops__", 0.0))
    if is_sparse and self_flops:
        self_flops *= _sparse_factor(module)

    total = self_flops
    for child in module.children():
        total += get_module_flops(child, is_sparse=is_sparse)
    return total


def get_module_macs(module: nn.Module, is_sparse: bool = False) -> float:
    """Recursively sum `__macs__` across a module hierarchy.

    Args:
        module: A `torch.nn.Module` whose submodules may define `__macs__`.
        is_sparse: If `True`, scale each module's self MACs by the sparsity ratio.

    Returns:
        Total MACs as a float.
    """
    self_macs = float(getattr(module, "__macs__", 0.0))
    if is_sparse and self_macs:
        self_macs *= _sparse_factor(module)

    total = self_macs
    for child in module.children():
        total += get_module_macs(child, is_sparse=is_sparse)
    return total


# --- Misc ---------------------------------------------------------------------

def convert_bytes(size: Union[int, float]) -> str:
    """Convert bytes to the largest binary unit (KB, MB, GB, TB, PB).

    Args:
        size: Byte count.

    Returns:
        A short string like '512 B', '1.23 KB', '4.56 MB', etc.

    Raises:
        ValueError: If `size` is negative.
    """
    if size < 0:
        raise ValueError("`size` must be non-negative.")
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    value = float(size)
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    s = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{s} {units[idx]}"


def is_package_available(pkg_name: str) -> bool:
    """Check whether a real importable package named `pkg_name` exists.

    Verifies both that an import spec can be found and that package metadata is
    available, to avoid false positives on bare directories named like a module.

    Args:
        pkg_name: Name of the package (e.g., 'torch', 'transformers').

    Returns:
        True if the package is importable and has metadata; False otherwise.
    """
    if importlib.util.find_spec(pkg_name) is None:
        return False
    try:
        # Accessing metadata confirms it's a real installed distribution.
        _ = importlib_metadata.metadata(pkg_name)
        return True
    except importlib_metadata.PackageNotFoundError:
        return False


# Backward-compat alias (original function name began with underscore).
_is_package_available = is_package_available
