from __future__ import annotations

import math
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prod(dims: Sequence[int]) -> int:
    """Integer product with empty-safe behavior (prod([])=1)."""
    p = 1
    for v in dims:
        p *= int(v)
    return p


def _as_tuple(val: Any, length: int) -> Tuple[int, ...]:
    """Convert scalar/int-or-seq to tuple of given length."""
    if isinstance(val, (tuple, list)):
        if len(val) != length:
            raise ValueError(
                f"Expected length {length}, got {len(val)} for {val}.")
        return tuple(int(v) for v in val)
    return tuple(int(val) for _ in range(length))


def _broadcast_shape(a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
    """Right-aligned broadcasting shape (NumPy/PyTorch semantics)."""
    ra, rb = list(reversed(a)), list(reversed(b))
    out: List[int] = []
    for i in range(max(len(ra), len(rb))):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da == 1:
            out.append(db)
        elif db == 1 or da == db:
            out.append(da)
        else:
            # 不可廣播時，回退為直接乘較大值（保守估算），避免中斷。
            out.append(max(da, db))
    return tuple(reversed(out))


def _conv_output_dims(
    input_dims: Sequence[int],
    kernel: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> Tuple[int, ...]:
    """Standard conv output spatial dims (floor division)."""
    out = []
    for i, s in enumerate(input_dims):
        k = kernel[i]
        st = stride[i]
        pad = padding[i]
        dil = dilation[i]
        o = (s + 2 * pad - (dil * (k - 1) + 1)) // st + 1
        out.append(int(o))
    return tuple(out)


def _conv_transpose_output_dims(
    input_dims: Sequence[int],
    kernel: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    output_padding: Sequence[int],
) -> Tuple[int, ...]:
    """Transposed conv output spatial dims."""
    out = []
    for i, s in enumerate(input_dims):
        k = kernel[i]
        st = stride[i]
        pad = padding[i]
        dil = dilation[i]
        op = output_padding[i]
        o = (s - 1) * st - 2 * pad + dil * (k - 1) + op + 1
        out.append(int(o))
    return tuple(out)


# ---------------------------------------------------------------------------
# FLOPs/MACs estimators for ops
# ---------------------------------------------------------------------------

def _linear_flops_compute(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tuple[int, int]:
    """FLOPs/MACs for linear: Y = X W^T + b, where input shape (*, in_features)."""
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    num_instances = int(input.numel()) // in_features  # number of rows/vectors
    macs = num_instances * in_features * out_features
    flops = 2 * macs + \
        (num_instances * out_features if bias is not None else 0)
    return int(flops), int(macs)


# Activations: count ~1 flop per element; MACs considered 0.

def _relu_flops_compute(input: Tensor, inplace: bool = False) -> Tuple[int, int]:
    return input.numel(), 0


def _prelu_flops_compute(input: Tensor, weight: Tensor) -> Tuple[int, int]:
    return input.numel(), 0


def _elu_flops_compute(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tuple[int, int]:
    return input.numel(), 0


def _leaky_relu_flops_compute(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tuple[int, int]:
    return input.numel(), 0


def _relu6_flops_compute(input: Tensor, inplace: bool = False) -> Tuple[int, int]:
    return input.numel(), 0


def _silu_flops_compute(input: Tensor, inplace: bool = False) -> Tuple[int, int]:
    return input.numel(), 0


def _gelu_flops_compute(input: Tensor, **kwargs: Any) -> Tuple[int, int]:
    return input.numel(), 0


def _pool_flops_compute(
    input: Tensor,
    kernel_size: Any,
    stride: Optional[Any] = None,
    padding: Any = 0,
    dilation: Optional[Any] = None,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    return_indices: Optional[bool] = None,
) -> Tuple[int, int]:
    """Use output elements count as FLOPs proxy (1 per output)."""
    spatial_dims = list(input.shape[2:])
    length = len(spatial_dims)
    k = _as_tuple(kernel_size, length)
    st = _as_tuple(stride if stride is not None else 1, length)
    pad = _as_tuple(padding, length)
    dil = _as_tuple(dilation if dilation is not None else 1, length)
    out_spatial = _conv_output_dims(spatial_dims, k, st, pad, dil)
    out_elems = int(input.shape[0]) * int(input.shape[1]) * _prod(out_spatial)
    return out_elems, 0


def _adaptive_pool_flops_compute(input: Tensor, output_size, *args, **kwargs) -> Tuple[int, int]:
    """FLOPs for adaptive pools: use output elements count directly.
    PyTorch adaptive pools接受 `output_size`（而非 kernel/stride/padding）。
    """
    # NCHW(d...) => 空間維度數
    d = input.dim() - 2
    if d <= 0:
        return 0, 0
    if isinstance(output_size, (tuple, list)):
        if len(output_size) == 1 and d > 1:
            out_spatial = (int(output_size[0]),) * d
        else:
            if len(output_size) != d:
                raise ValueError(
                    f"adaptive output_size length {len(output_size)} != spatial dims {d}")
            out_spatial = tuple(int(v) for v in output_size)
    else:
        out_spatial = (int(output_size),) * d

    out_elems = int(input.shape[0]) * int(input.shape[1]) * _prod(out_spatial)
    return out_elems, 0


def _conv_flops_compute(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tuple[int, int]:
    """FLOPs/MACs for standard N-D convolution (NCHW-d)."""
    assert int(weight.shape[1]) * int(groups) == int(input.shape[1]), (
        "Conv weight/input channels mismatch with groups."
    )
    batch = int(input.shape[0])
    in_c = int(input.shape[1])
    out_c = int(weight.shape[0])
    kernel = [int(x) for x in weight.shape[2:]]
    in_spatial = [int(x) for x in input.shape[2:]]
    d = len(in_spatial)

    st = _as_tuple(stride, d)
    if isinstance(padding, str):
        if padding.lower() == "valid":
            pad = tuple(0 for _ in range(d))
        elif padding.lower() == "same":
            # approximate 'same' padding for stride=1, common in HF models
            pad = tuple(((kd - 1) // 2) for kd in kernel)
        else:
            raise ValueError(f"Unsupported padding string: {padding}")
    else:
        pad = _as_tuple(padding, d)
    dil = _as_tuple(dilation, d)

    out_spatial = _conv_output_dims(in_spatial, kernel, st, pad, dil)
    elements = batch * _prod(out_spatial)
    conv_per_pos_macs = (in_c // groups) * _prod(kernel) * out_c
    macs = elements * conv_per_pos_macs
    flops = 2 * macs + (elements * out_c if bias is not None else 0)
    return int(flops), int(macs)


def _conv_trans_flops_compute(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    groups: int = 1,
    dilation: Any = 1,
) -> Tuple[int, int]:
    """FLOPs/MACs for transposed convolution (a.k.a. deconvolution)."""
    batch = int(input.shape[0])
    in_c = int(input.shape[1])
    # PyTorch weight shape for convT: (in_channels, out_channels // groups, *k)
    out_c = int(weight.shape[1])
    kernel = [int(x) for x in weight.shape[2:]]
    in_spatial = [int(x) for x in input.shape[2:]]
    d = len(in_spatial)

    st = _as_tuple(stride, d)
    pad = _as_tuple(padding, d)
    dil = _as_tuple(dilation, d)
    op = _as_tuple(output_padding, d)

    # 正確的 output dims（僅用於 bias FLOPs）
    out_spatial = _conv_transpose_output_dims(
        in_spatial, kernel, st, pad, dil, op)

    # MACs：每個輸入位置會以 kernel 擴散，等價於：in_elems * (out_c/groups) * kernel_area
    in_elements = batch * _prod(in_spatial)
    conv_per_pos_macs = (out_c // groups) * _prod(kernel) * in_c
    macs = in_elements * conv_per_pos_macs
    flops = 2 * macs + (batch * out_c * _prod(out_spatial)
                        if bias is not None else 0)
    return int(flops), int(macs)


def _batch_norm_flops_compute(
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> Tuple[int, int]:
    has_affine = weight is not None
    if training:
        return input.numel() * (5 if has_affine else 4), 0
    return input.numel() * (2 if has_affine else 1), 0


def _layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[int, int]:
    has_affine = weight is not None
    return input.numel() * (5 if has_affine else 4), 0


def _group_norm_flops_compute(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[int, int]:
    has_affine = weight is not None
    return input.numel() * (5 if has_affine else 4), 0


def _instance_norm_flops_compute(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tuple[int, int]:
    has_affine = weight is not None
    return input.numel() * (5 if has_affine else 4), 0


def _upsample_flops_compute(*args: Any, **kwargs: Any) -> Tuple[int, int]:
    """FLOPs for upsample/interpolate: 以輸出元素數作為近似 FLOPs。
    支援以 `size` 或 `scale_factor` 指定。
    """
    input: Tensor = args[0]
    size = kwargs.get("size", None)
    if size is None and len(args) > 1:
        size = args[1]

    spatial_in = list(input.shape[2:])
    d = len(spatial_in)

    if size is not None:
        if isinstance(size, (tuple, list)):
            out_spatial = tuple(int(v) for v in size)
            if len(out_spatial) != d:
                # 允許單一整數時複製到所有空間維度
                if len(out_spatial) == 1:
                    out_spatial = (out_spatial[0],) * d
                else:
                    raise ValueError(
                        f"size length {len(out_spatial)} mismatches spatial dims {d}")
        else:
            out_spatial = (int(size),) * d
    else:
        scale = kwargs.get("scale_factor", None)
        if scale is None and len(args) > 2:
            scale = args[2]
        if scale is None:
            raise AssertionError(
                "either size or scale_factor should be defined")
        if isinstance(scale, (tuple, list)):
            if len(scale) != d:
                if len(scale) == 1:
                    scale = (float(scale[0]),) * d
                else:
                    raise ValueError(
                        f"scale_factor length {len(scale)} mismatches spatial dims {d}")
            scale_f = [float(s) for s in scale]
        else:
            scale_f = [float(scale)] * d
        out_spatial = tuple(
            int(math.floor(spatial_in[i] * scale_f[i])) for i in range(d))

    out_elems = int(input.shape[0]) * int(input.shape[1]) * _prod(out_spatial)
    return out_elems, 0


def _softmax_flops_compute(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[torch.dtype] = None) -> Tuple[int, int]:
    return input.numel(), 0


def _embedding_flops_compute(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tuple[int, int]:
    return 0, 0


def _dropout_flops_compute(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tuple[int, int]:
    return 0, 0


def _matmul_flops_compute(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tuple[int, int]:
    """Generic batched matmul FLOPs/MACs: (..., M, K) x (..., K, N) -> (..., M, N)."""
    if input.dim() < 1 or other.dim() < 1:
        return 0, 0
    M = int(input.shape[-2]) if input.dim() >= 2 else 1
    K1 = int(input.shape[-1])
    K2 = int(other.shape[-2]) if other.dim() >= 2 else int(other.shape[-1])
    N = int(other.shape[-1]) if other.dim() >= 2 else 1
    # 防禦性處理：K 不一致時仍以較小者估算
    K = min(K1, K2)
    batch_shape = _broadcast_shape(input.shape[:-2] if input.dim() >= 2 else (1,),
                                   other.shape[:-2] if other.dim() >= 2 else (1,))
    batch_elems = _prod(batch_shape if batch_shape else (1,))
    macs = batch_elems * M * K * N
    return 2 * macs, macs


def _addmm_flops_compute(input: Tensor, mat1: Tensor, mat2: Tensor, *, beta: float = 1, alpha: float = 1, out: Optional[Tensor] = None) -> Tuple[int, int]:
    M, K = int(mat1.shape[-2]), int(mat1.shape[-1])
    K2, N = int(mat2.shape[-2]), int(mat2.shape[-1])
    K_use = min(K, K2)
    macs = M * K_use * N
    add_elems = _prod(input.shape)
    flops = 2 * macs + add_elems  # 忽略 alpha/beta 是否為 0 的情況
    return flops, macs


def _baddbmm_flops_compute(input: Tensor, batch1: Tensor, batch2: Tensor, *, beta: float = 1, alpha: float = 1, out: Optional[Tensor] = None) -> Tuple[int, int]:
    """Batched addbmm: input (b,M,N) + batch1(b,M,K) @ batch2(b,K,N)."""
    b = int(batch1.shape[0]) if batch1.dim() > 2 else 1
    M, K = int(batch1.shape[-2]), int(batch1.shape[-1])
    K2, N = int(batch2.shape[-2]), int(batch2.shape[-1])
    K_use = min(K, K2)
    macs = b * M * K_use * N
    add_elems = _prod(input.shape)
    flops = 2 * macs + add_elems
    return flops, macs


def _einsum_flops_compute(equation: str, *operands: Tensor) -> Tuple[int, int]:
    """Estimate FLOPs for einsum via NumPy's path report; fallback NotImplemented."""
    eq = equation.replace(" ", "")
    input_shapes = [tuple(int(x) for x in o.shape) for o in operands]
    # Re-map to canonical letters
    letter_order = OrderedDict((k, 0) for k in eq if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    eq = eq.translate(mapping)
    try:
        np_arrs = [np.zeros(s, dtype=np.uint8) for s in input_shapes]
        path_info = np.einsum_path(eq, *np_arrs, optimize="optimal")[1]
        for line in path_info.split("\n"):
            if "flop" in line.lower():
                flop = int(float(line.split(":")[-1]))
                return flop, 0
    except MemoryError:
        pass
    raise NotImplementedError("Unsupported or too-large einsum operation.")


def _tensor_addmm_flops_compute(self: Tensor, mat1: Tensor, mat2: Tensor, *, beta: float = 1, alpha: float = 1, out: Optional[Tensor] = None) -> Tuple[int, int]:
    M, K = int(mat1.shape[-2]), int(mat1.shape[-1])
    K2, N = int(mat2.shape[-2]), int(mat2.shape[-1])
    K_use = min(K, K2)
    macs = M * K_use * N
    add_elems = _prod(self.shape)
    flops = 2 * macs + add_elems
    return flops, macs


def _mul_flops_compute(input: Any, other: Any, *, out: Optional[Tensor] = None) -> Tuple[int, int]:
    return _elementwise_flops_compute(input, other)


def _add_flops_compute(input: Any, other: Any, *, alpha: float = 1, out: Optional[Tensor] = None) -> Tuple[int, int]:
    return _elementwise_flops_compute(input, other)


def _elementwise_flops_compute(input: Any, other: Any) -> Tuple[int, int]:
    """Elementwise/add/mul FLOPs：以廣播後形狀元素數估計。"""
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape), 0
        return 1, 0
    if not torch.is_tensor(other):
        return _prod(input.shape), 0
    final_shape = _broadcast_shape(
        tuple(int(x) for x in input.shape), tuple(int(x) for x in other.shape))
    return _prod(final_shape), 0


# ---------------------------------------------------------------------------
# RNN hooks
# ---------------------------------------------------------------------------

def _rnn_flops(flops: int, rnn_module: nn.Module, w_ih: Tensor, w_hh: Tensor, input_size: int) -> int:
    gates_size = int(w_ih.shape[0])
    # ih 和 hh 的矩陣乘
    flops += 2 * int(w_ih.shape[0]) * int(w_ih.shape[1]) - gates_size
    flops += 2 * int(w_hh.shape[0]) * int(w_hh.shape[1]) - gates_size
    hs = int(getattr(rnn_module, "hidden_size", gates_size))
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        flops += hs
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        flops += hs  # hadamard of r
        flops += hs * 3  # two states add
        flops += hs * 3  # last hadamard+add
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        flops += hs * 4
        flops += hs + hs + hs  # two hadamard + add (C state)
        flops += hs + hs + hs  # final hadamard
    return flops


def _rnn_forward_hook(rnn_module: nn.Module, input: Tuple[Tensor, ...], output: Any) -> None:
    flops = 0
    inp = input[0]
    batch_first = bool(getattr(rnn_module, "batch_first", False))
    if batch_first:
        batch_size, seq_length = int(inp.shape[0]), int(inp.shape[1])
    else:
        seq_length, batch_size = int(inp.shape[0]), int(inp.shape[1])

    num_layers = int(getattr(rnn_module, "num_layers", 1))

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__(f"weight_ih_l{i}")
        w_hh = rnn_module.__getattr__(f"weight_hh_l{i}")
        input_size = int(getattr(rnn_module, "input_size", w_ih.shape[1])) if i == 0 else int(
            getattr(rnn_module, "hidden_size", w_ih.shape[1])
        )
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if bool(getattr(rnn_module, "bias", False)):
            b_ih = rnn_module.__getattr__(f"bias_ih_l{i}")
            b_hh = rnn_module.__getattr__(f"bias_hh_l{i}")
            flops += int(b_ih.shape[0]) + int(b_hh.shape[0])

    flops *= batch_size
    flops *= seq_length
    if bool(getattr(rnn_module, "bidirectional", False)):
        flops *= 2
    rnn_module.__flops__ = int(
        getattr(rnn_module, "__flops__", 0)) + int(flops)


def _rnn_cell_forward_hook(rnn_cell_module: nn.Module, input: Tuple[Tensor, ...], output: Any) -> None:
    flops = 0
    inp = input[0]
    batch_size = int(inp.shape[0])
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = int(inp.shape[1])
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if bool(getattr(rnn_cell_module, "bias", False)):
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += int(b_ih.shape[0]) + int(b_hh.shape[0])

    flops *= batch_size
    rnn_cell_module.__flops__ = int(
        getattr(rnn_cell_module, "__flops__", 0)) + int(flops)


MODULE_HOOK_MAPPING: Dict[Any, Callable[..., None]] = {
    nn.RNN: _rnn_forward_hook,
    nn.GRU: _rnn_forward_hook,
    nn.LSTM: _rnn_forward_hook,
    nn.RNNCell: _rnn_cell_forward_hook,
    nn.LSTMCell: _rnn_cell_forward_hook,
    nn.GRUCell: _rnn_cell_forward_hook,
}

# ---------------------------------------------------------------------------
# Monkey-patching helpers (safe)
# ---------------------------------------------------------------------------


def wrapFunc(
    func: Callable[..., Any],
    funcFlopCompute: Callable[..., Tuple[int, int]],
    old_functions: Dict[str, Callable[..., Any]],
    module_flop_count: Optional[List[List[Tuple[str, int]]]],
    module_mac_count: Optional[List[List[Tuple[str, int]]]],
    qualname: Optional[str] = None,
) -> Callable[..., Any]:
    """Wrap a function to record FLOPs/MACs and call the original.

    Args:
      func: Original function to be wrapped.
      funcFlopCompute: FLOPs/MACs estimation function with the same signature.
      old_functions: Registry dict; original function will be saved here by key.
      module_flop_count: A stack (list) of lists to append (op_name, flops).
      module_mac_count: A stack (list) of lists to append (op_name, macs).
      qualname: Dotted path key; if None, derive from func.

    Returns:
      new_func: The wrapped callable.
    """
    key = qualname or f"{getattr(func, '__module__', '')}.{getattr(func, '__name__', str(func))}"
    if key not in old_functions:
        old_functions[key] = func

    @wraps(func)
    def newFunc(*args: Any, **kwds: Any) -> Any:
        try:
            flops, macs = funcFlopCompute(*args, **kwds)
        except Exception:
            flops, macs = 0, 0
        if module_flop_count is not None and len(module_flop_count) > 0:
            module_flop_count[-1].append((key, int(flops)))
        if module_mac_count is not None and len(module_mac_count) > 0:
            module_mac_count[-1].append((key, int(macs)))
        return old_functions[key](*args, **kwds)

    return newFunc


def _patch_functionals(old_functions: Dict[str, Callable[..., Any]], module_flop_count: Optional[List[List[Tuple[str, int]]]], module_mac_count: Optional[List[List[Tuple[str, int]]]]) -> None:
    """Monkey-patch torch.nn.functional APIs for FLOPs/MACs logging."""
    # Linear & conv
    F.linear = wrapFunc(F.linear, _linear_flops_compute, old_functions,
                        module_flop_count, module_mac_count, "torch.nn.functional.linear")
    F.conv1d = wrapFunc(F.conv1d, _conv_flops_compute, old_functions,
                        module_flop_count, module_mac_count, "torch.nn.functional.conv1d")
    F.conv2d = wrapFunc(F.conv2d, _conv_flops_compute, old_functions,
                        module_flop_count, module_mac_count, "torch.nn.functional.conv2d")
    F.conv3d = wrapFunc(F.conv3d, _conv_flops_compute, old_functions,
                        module_flop_count, module_mac_count, "torch.nn.functional.conv3d")

    # Conv transpose
    F.conv_transpose1d = wrapFunc(F.conv_transpose1d, _conv_trans_flops_compute, old_functions,
                                  module_flop_count, module_mac_count, "torch.nn.functional.conv_transpose1d")
    F.conv_transpose2d = wrapFunc(F.conv_transpose2d, _conv_trans_flops_compute, old_functions,
                                  module_flop_count, module_mac_count, "torch.nn.functional.conv_transpose2d")
    F.conv_transpose3d = wrapFunc(F.conv_transpose3d, _conv_trans_flops_compute, old_functions,
                                  module_flop_count, module_mac_count, "torch.nn.functional.conv_transpose3d")

    # Activations
    F.relu = wrapFunc(F.relu, _relu_flops_compute, old_functions,
                      module_flop_count, module_mac_count, "torch.nn.functional.relu")
    F.prelu = wrapFunc(F.prelu, _prelu_flops_compute, old_functions,
                       module_flop_count, module_mac_count, "torch.nn.functional.prelu")
    F.elu = wrapFunc(F.elu, _elu_flops_compute, old_functions,
                     module_flop_count, module_mac_count, "torch.nn.functional.elu")
    F.leaky_relu = wrapFunc(F.leaky_relu, _leaky_relu_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.leaky_relu")
    F.relu6 = wrapFunc(F.relu6, _relu6_flops_compute, old_functions,
                       module_flop_count, module_mac_count, "torch.nn.functional.relu6")
    if hasattr(F, "silu"):
        F.silu = wrapFunc(F.silu, _silu_flops_compute, old_functions,
                          module_flop_count, module_mac_count, "torch.nn.functional.silu")
    F.gelu = wrapFunc(F.gelu, _gelu_flops_compute, old_functions,
                      module_flop_count, module_mac_count, "torch.nn.functional.gelu")

    # Normalizations
    F.batch_norm = wrapFunc(F.batch_norm, _batch_norm_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.batch_norm")
    F.layer_norm = wrapFunc(F.layer_norm, _layer_norm_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.layer_norm")
    F.instance_norm = wrapFunc(F.instance_norm, _instance_norm_flops_compute, old_functions,
                               module_flop_count, module_mac_count, "torch.nn.functional.instance_norm")
    F.group_norm = wrapFunc(F.group_norm, _group_norm_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.group_norm")

    # Poolings
    F.avg_pool1d = wrapFunc(F.avg_pool1d, _pool_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.avg_pool1d")
    F.avg_pool2d = wrapFunc(F.avg_pool2d, _pool_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.avg_pool2d")
    F.avg_pool3d = wrapFunc(F.avg_pool3d, _pool_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.avg_pool3d")
    F.max_pool1d = wrapFunc(F.max_pool1d, _pool_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.max_pool1d")
    F.max_pool2d = wrapFunc(F.max_pool2d, _pool_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.max_pool2d")
    F.max_pool3d = wrapFunc(F.max_pool3d, _pool_flops_compute, old_functions,
                            module_flop_count, module_mac_count, "torch.nn.functional.max_pool3d")
    F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, _adaptive_pool_flops_compute, old_functions,
                                     module_flop_count, module_mac_count, "torch.nn.functional.adaptive_avg_pool1d")
    F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, _adaptive_pool_flops_compute, old_functions,
                                     module_flop_count, module_mac_count, "torch.nn.functional.adaptive_avg_pool2d")
    F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, _adaptive_pool_flops_compute, old_functions,
                                     module_flop_count, module_mac_count, "torch.nn.functional.adaptive_avg_pool3d")
    F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, _adaptive_pool_flops_compute, old_functions,
                                     module_flop_count, module_mac_count, "torch.nn.functional.adaptive_max_pool1d")
    F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, _adaptive_pool_flops_compute, old_functions,
                                     module_flop_count, module_mac_count, "torch.nn.functional.adaptive_max_pool2d")
    F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, _adaptive_pool_flops_compute, old_functions,
                                     module_flop_count, module_mac_count, "torch.nn.functional.adaptive_max_pool3d")

    # Upsample / interpolate
    F.upsample = wrapFunc(F.upsample, _upsample_flops_compute, old_functions,
                          module_flop_count, module_mac_count, "torch.nn.functional.upsample")
    F.interpolate = wrapFunc(F.interpolate, _upsample_flops_compute, old_functions,
                             module_flop_count, module_mac_count, "torch.nn.functional.interpolate")

    # Softmax & embedding & dropout
    F.softmax = wrapFunc(F.softmax, _softmax_flops_compute, old_functions,
                         module_flop_count, module_mac_count, "torch.nn.functional.softmax")
    F.embedding = wrapFunc(F.embedding, _embedding_flops_compute, old_functions,
                           module_flop_count, module_mac_count, "torch.nn.functional.embedding")


def _patch_tensor_methods(old_functions: Dict[str, Callable[..., Any]], module_flop_count: Optional[List[List[Tuple[str, int]]]], module_mac_count: Optional[List[List[Tuple[str, int]]]]) -> None:
    """Monkey-patch selected torch / Tensor methods."""
    torch.matmul = wrapFunc(torch.matmul, _matmul_flops_compute,
                            old_functions, module_flop_count, module_mac_count, "torch.matmul")
    torch.Tensor.matmul = wrapFunc(torch.Tensor.matmul, _matmul_flops_compute,
                                   old_functions, module_flop_count, module_mac_count, "torch.Tensor.matmul")

    torch.addmm = wrapFunc(torch.addmm, _addmm_flops_compute, old_functions,
                           module_flop_count, module_mac_count, "torch.addmm")
    torch.Tensor.addmm = wrapFunc(torch.Tensor.addmm, _tensor_addmm_flops_compute,
                                  old_functions, module_flop_count, module_mac_count, "torch.Tensor.addmm")

    torch.mul = wrapFunc(torch.mul, _mul_flops_compute, old_functions,
                         module_flop_count, module_mac_count, "torch.mul")
    torch.Tensor.mul = wrapFunc(torch.Tensor.mul, _mul_flops_compute,
                                old_functions, module_flop_count, module_mac_count, "torch.Tensor.mul")

    torch.add = wrapFunc(torch.add, _add_flops_compute, old_functions,
                         module_flop_count, module_mac_count, "torch.add")
    torch.Tensor.add = wrapFunc(torch.Tensor.add, _add_flops_compute,
                                old_functions, module_flop_count, module_mac_count, "torch.Tensor.add")

    torch.einsum = wrapFunc(torch.einsum, _einsum_flops_compute,
                            old_functions, module_flop_count, module_mac_count, "torch.einsum")

    # 專門處理 batched addbmm
    torch.baddbmm = wrapFunc(torch.baddbmm, _baddbmm_flops_compute,
                             old_functions, module_flop_count, module_mac_count, "torch.baddbmm")


def _reload_functionals(old_functions: Dict[str, Callable[..., Any]]) -> None:
    """Restore original torch.nn.functional functions from registry."""
    keys = [
        "torch.nn.functional.linear",
        "torch.nn.functional.conv1d",
        "torch.nn.functional.conv2d",
        "torch.nn.functional.conv3d",
        "torch.nn.functional.conv_transpose1d",
        "torch.nn.functional.conv_transpose2d",
        "torch.nn.functional.conv_transpose3d",
        "torch.nn.functional.relu",
        "torch.nn.functional.prelu",
        "torch.nn.functional.elu",
        "torch.nn.functional.leaky_relu",
        "torch.nn.functional.relu6",
        "torch.nn.functional.silu",
        "torch.nn.functional.gelu",
        "torch.nn.functional.batch_norm",
        "torch.nn.functional.layer_norm",
        "torch.nn.functional.instance_norm",
        "torch.nn.functional.group_norm",
        "torch.nn.functional.avg_pool1d",
        "torch.nn.functional.avg_pool2d",
        "torch.nn.functional.avg_pool3d",
        "torch.nn.functional.max_pool1d",
        "torch.nn.functional.max_pool2d",
        "torch.nn.functional.max_pool3d",
        "torch.nn.functional.adaptive_avg_pool1d",
        "torch.nn.functional.adaptive_avg_pool2d",
        "torch.nn.functional.adaptive_avg_pool3d",
        "torch.nn.functional.adaptive_max_pool1d",
        "torch.nn.functional.adaptive_max_pool2d",
        "torch.nn.functional.adaptive_max_pool3d",
        "torch.nn.functional.upsample",
        "torch.nn.functional.interpolate",
        "torch.nn.functional.softmax",
        "torch.nn.functional.embedding",
    ]
    for k in keys:
        if k in old_functions:
            module_name, attr = k.rsplit(".", 1)
            module = {"torch.nn.functional": F}.get(module_name, None)
            if module is not None and hasattr(module, attr):
                setattr(module, attr, old_functions[k])


def _reload_tensor_methods(old_functions: Dict[str, Callable[..., Any]]) -> None:
    keys = [
        "torch.matmul",
        "torch.Tensor.matmul",
        "torch.addmm",
        "torch.Tensor.addmm",
        "torch.mul",
        "torch.Tensor.mul",
        "torch.add",
        "torch.Tensor.add",
        "torch.einsum",
        "torch.baddbmm",
    ]
    for k in keys:
        if k in old_functions:
            if k == "torch.Tensor.matmul":
                torch.Tensor.matmul = old_functions[k]
            elif k == "torch.Tensor.addmm":
                torch.Tensor.addmm = old_functions[k]
            elif k == "torch.Tensor.mul":
                torch.Tensor.mul = old_functions[k]
            elif k == "torch.Tensor.add":
                torch.Tensor.add = old_functions[k]
            else:
                # torch module level
                mod_name, attr = k.rsplit(".", 1)
                if mod_name == "torch" and hasattr(torch, attr):
                    setattr(torch, attr, old_functions[k])
