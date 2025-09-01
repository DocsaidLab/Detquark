from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ops = importlib.import_module("utils.calflops.pytorch_ops")


# -----------------------------------------------------------------------------
# Pytest fixtures for patching & restoring
# -----------------------------------------------------------------------------
@pytest.fixture
def patched_ops():
    """Patch torch/torch.nn.functional and restore them after the test."""
    old: Dict[str, callable] = {}
    flops_stack: List[List[Tuple[str, int]]] = [
        []]  # stack top for collecting (op, flops)
    macs_stack: List[List[Tuple[str, int]]] = [
        []]   # stack top for collecting (op, macs)

    # record original ids for idempotency check
    orig_ids = {
        "linear": id(F.linear),
        "conv2d": id(F.conv2d),
        "interpolate": id(F.interpolate),
        "matmul": id(torch.matmul),
        "addmm": id(torch.addmm),
    }

    ops._patch_functionals(old, flops_stack, macs_stack)
    ops._patch_tensor_methods(old, flops_stack, macs_stack)

    try:
        yield {
            "old": old,
            "flops_stack": flops_stack,
            "macs_stack": macs_stack,
            "orig_ids": orig_ids,
        }
    finally:
        ops._reload_functionals(old)
        ops._reload_tensor_methods(old)


def _take_records(ctx, key: str | None = None):
    """Helper: get and clear the top-of-stack records. Optionally filter by key."""
    fl = ctx["flops_stack"][-1]
    mc = ctx["macs_stack"][-1]
    fl_out = fl[:] if key is None else [v for (k, v) in fl if k == key]
    mc_out = mc[:] if key is None else [v for (k, v) in mc if k == key]
    fl.clear()
    mc.clear()
    return fl_out, mc_out


# -----------------------------------------------------------------------------
# Basic patching invariants
# -----------------------------------------------------------------------------
def test_patch_and_reload_idempotent(patched_ops):
    ctx = patched_ops
    # After patch, the function object should differ
    assert id(F.linear) != ctx["orig_ids"]["linear"]
    assert id(F.conv2d) != ctx["orig_ids"]["conv2d"]
    assert id(F.interpolate) != ctx["orig_ids"]["interpolate"]
    assert id(torch.matmul) != ctx["orig_ids"]["matmul"]
    assert id(torch.addmm) != ctx["orig_ids"]["addmm"]

    # After reload, it should be back to original
    ops._reload_functionals(ctx["old"])
    ops._reload_tensor_methods(ctx["old"])
    assert id(F.linear) == ctx["orig_ids"]["linear"]
    assert id(F.conv2d) == ctx["orig_ids"]["conv2d"]
    assert id(F.interpolate) == ctx["orig_ids"]["interpolate"]
    assert id(torch.matmul) == ctx["orig_ids"]["matmul"]
    assert id(torch.addmm) == ctx["orig_ids"]["addmm"]


# -----------------------------------------------------------------------------
# Activations (FIXED: parametrize only names; fetch functions after patch)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "act_name",
    [
        "relu",
        "elu",
        "leaky_relu",
        "gelu",
        "prelu",
        pytest.param(
            "relu6",
            marks=pytest.mark.skipif(not hasattr(
                F, "relu6"), reason="relu6 not available in this torch build"),
        ),
        pytest.param(
            "silu",
            marks=pytest.mark.skipif(not hasattr(
                F, "silu"), reason="silu not available in this torch build"),
        ),
    ],
)
def test_activation_flops_are_numel_and_macs_zero(patched_ops, act_name):
    ctx = patched_ops
    x = torch.randn(2, 3, 5, 7)

    act = getattr(F, act_name)
    if act_name == "prelu":
        w = torch.ones(x.shape[1])  # channel-wise
        _ = act(x, w)
    elif act_name == "leaky_relu":
        _ = act(x, negative_slope=0.2)
    else:
        _ = act(x)

    fl, mc = _take_records(ctx, f"torch.nn.functional.{act_name}")
    assert len(fl) == 1 and len(mc) == 1
    assert fl[0] == x.numel()
    assert mc[0] == 0


# -----------------------------------------------------------------------------
# Convolution (forward)
# -----------------------------------------------------------------------------
def test_conv2d_flops_macs_with_bias(patched_ops):
    ctx = patched_ops
    N, Cin, H, W = 2, 3, 32, 32
    Cout, k = 8, 3
    x = torch.randn(N, Cin, H, W)
    w = torch.randn(Cout, Cin, k, k)
    b = torch.randn(Cout)
    y = F.conv2d(x, w, b, stride=1, padding=1, dilation=1, groups=1)
    assert y.shape == (N, Cout, H, W)

    fl, mc = _take_records(ctx, "torch.nn.functional.conv2d")
    elements = N * H * W
    conv_per_pos_macs = (Cin // 1) * (k * k) * Cout
    macs = elements * conv_per_pos_macs
    flops = 2 * macs + (elements * Cout)  # bias
    assert mc[0] == macs
    assert fl[0] == flops


def test_conv2d_groups_and_stride_padding(patched_ops):
    ctx = patched_ops
    N, Cin, H, W = 1, 4, 15, 17
    groups = 2
    Cout, k = 6, 3
    x = torch.randn(N, Cin, H, W)
    w = torch.randn(Cout, Cin // groups, k, k)
    y = F.conv2d(x, w, None, stride=2, padding=1, dilation=1, groups=groups)
    Ho = (H + 2 * 1 - (1 * (k - 1) + 1)) // 2 + 1
    Wo = (W + 2 * 1 - (1 * (k - 1) + 1)) // 2 + 1
    assert y.shape == (N, Cout, Ho, Wo)

    fl, mc = _take_records(ctx, "torch.nn.functional.conv2d")
    elements = N * Ho * Wo
    conv_per_pos_macs = (Cin // groups) * (k * k) * Cout
    macs = elements * conv_per_pos_macs
    flops = 2 * macs  # no bias
    assert mc[0] == macs
    assert fl[0] == flops


# -----------------------------------------------------------------------------
# ConvTranspose
# -----------------------------------------------------------------------------
def test_conv_transpose2d_flops_bias_and_output_dims(patched_ops):
    ctx = patched_ops
    N, Cin, Hin, Win = 2, 8, 16, 16
    Cout, k = 4, 3
    # convT weight shape (in_c, out_c // groups, kH, kW)
    w = torch.randn(Cin, Cout, k, k)
    b = torch.randn(Cout)
    x = torch.randn(N, Cin, Hin, Win)
    stride, padding, dilation, output_padding, groups = (
        2, 2), (1, 1), (1, 1), (1, 1), 1

    y = F.conv_transpose2d(x, w, b, stride=stride, padding=padding,
                           dilation=dilation, output_padding=output_padding, groups=groups)
    Hout = (Hin - 1) * stride[0] - 2 * padding[0] + \
        dilation[0] * (k - 1) + output_padding[0] + 1
    Wout = (Win - 1) * stride[1] - 2 * padding[1] + \
        dilation[1] * (k - 1) + output_padding[1] + 1
    assert y.shape == (N, Cout, Hout, Wout)

    fl, mc = _take_records(ctx, "torch.nn.functional.conv_transpose2d")
    in_elements = N * Hin * Win
    conv_per_pos_macs = (Cout // groups) * (k * k) * Cin
    macs = in_elements * conv_per_pos_macs
    flops = 2 * macs + (N * Cout * Hout * Wout)
    assert mc[0] == macs
    assert fl[0] == flops


# -----------------------------------------------------------------------------
# Pooling
# -----------------------------------------------------------------------------
def test_avg_pool2d_flops_equals_output_elements(patched_ops):
    ctx = patched_ops
    x = torch.randn(3, 5, 20, 22)
    y = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
    Ho = (20 + 2 * 1 - (1 * (3 - 1) + 1)) // 2 + 1
    Wo = (22 + 2 * 1 - (1 * (3 - 1) + 1)) // 2 + 1
    assert y.shape == (3, 5, Ho, Wo)

    fl, mc = _take_records(ctx, "torch.nn.functional.avg_pool2d")
    out_elems = 3 * 5 * Ho * Wo
    assert fl[0] == out_elems
    assert mc[0] == 0


# -----------------------------------------------------------------------------
# Upsample / Interpolate
# -----------------------------------------------------------------------------
def test_interpolate_with_size(patched_ops):
    ctx = patched_ops
    x = torch.randn(2, 3, 7, 9)
    y = F.interpolate(x, size=(14, 18), mode="nearest")
    assert y.shape == (2, 3, 14, 18)

    fl, mc = _take_records(ctx, "torch.nn.functional.interpolate")
    assert fl[0] == 2 * 3 * 14 * 18
    assert mc[0] == 0


def test_interpolate_with_scale_factor_integer(patched_ops):
    ctx = patched_ops
    x = torch.randn(1, 2, 8, 5)
    y = F.interpolate(x, scale_factor=2, mode="nearest")
    assert y.shape == (1, 2, 16, 10)

    fl, mc = _take_records(ctx, "torch.nn.functional.interpolate")
    assert fl[0] == 1 * 2 * 16 * 10
    assert mc[0] == 0


def test__upsample_flops_compute_raises_when_missing_params():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(AssertionError):
        ops._upsample_flops_compute(x)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Softmax / Embedding
# -----------------------------------------------------------------------------
def test_softmax_flops_numel(patched_ops):
    ctx = patched_ops
    x = torch.randn(4, 6, 7)
    _ = F.softmax(x, dim=-1)
    fl, mc = _take_records(ctx, "torch.nn.functional.softmax")
    assert fl[0] == x.numel()
    assert mc[0] == 0


def test_embedding_flops_zero(patched_ops):
    ctx = patched_ops
    weight = torch.randn(10, 6)
    idx = torch.tensor([[1, 2, 3], [4, 0, 5]])
    _ = F.embedding(idx, weight)
    fl, mc = _take_records(ctx, "torch.nn.functional.embedding")
    assert fl[0] == 0
    assert mc[0] == 0


# -----------------------------------------------------------------------------
# Elementwise + Broadcasting
# -----------------------------------------------------------------------------
def test_elementwise_add_broadcasting_counts_output_elems(patched_ops):
    ctx = patched_ops
    a = torch.randn(2, 3, 4)
    b = torch.randn(1, 3, 1)
    _ = torch.add(a, b)
    fl, mc = _take_records(ctx, "torch.add")
    assert fl[0] == 2 * 3 * 4
    assert mc[0] == 0


def test_elementwise_mul_mismatched_ranks_broadcast(patched_ops):
    ctx = patched_ops
    a = torch.randn(4, 3, 1)
    b = torch.randn(3, 5)
    _ = torch.mul(a, b)  # -> (4,3,5)
    fl, mc = _take_records(ctx, "torch.mul")
    assert fl[0] == 4 * 3 * 5
    assert mc[0] == 0


# -----------------------------------------------------------------------------
# Matmul / Addmm / Baddbmm
# -----------------------------------------------------------------------------
def test_torch_matmul_batched(patched_ops):
    ctx = patched_ops
    B, M, K, N = 3, 4, 5, 6
    a = torch.randn(B, M, K)
    b = torch.randn(B, K, N)
    _ = torch.matmul(a, b)
    fl, mc = _take_records(ctx, "torch.matmul")
    macs = B * M * K * N
    flops = 2 * macs
    assert mc[0] == macs
    assert fl[0] == flops


def test_torch_addmm(patched_ops):
    ctx = patched_ops
    M, K, N = 4, 5, 6
    inp = torch.randn(M, N)
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    _ = torch.addmm(inp, a, b)
    fl, mc = _take_records(ctx, "torch.addmm")
    macs = M * K * N
    flops = 2 * macs + inp.numel()
    assert mc[0] == macs
    assert fl[0] == flops


def test_torch_baddbmm(patched_ops):
    ctx = patched_ops
    B, M, K, N = 2, 3, 4, 5
    inp = torch.randn(B, M, N)
    a = torch.randn(B, M, K)
    b = torch.randn(B, K, N)
    _ = torch.baddbmm(inp, a, b)
    fl, mc = _take_records(ctx, "torch.baddbmm")
    macs = B * M * K * N
    flops = 2 * macs + inp.numel()
    assert mc[0] == macs
    assert fl[0] == flops


# -----------------------------------------------------------------------------
# Normalizations
# -----------------------------------------------------------------------------
def test_batch_norm_training_and_eval(patched_ops):
    ctx = patched_ops
    x = torch.randn(2, 4, 5, 6)
    running_mean = torch.zeros(4)
    running_var = torch.ones(4)
    weight = torch.ones(4)
    bias = torch.zeros(4)

    _ = F.batch_norm(x, running_mean, running_var,
                     weight=weight, bias=bias, training=True)
    fl, mc = _take_records(ctx, "torch.nn.functional.batch_norm")
    assert fl[0] == x.numel() * 5  # with affine
    assert mc[0] == 0

    _ = F.batch_norm(x, running_mean, running_var,
                     weight=weight, bias=bias, training=False)
    fl, mc = _take_records(ctx, "torch.nn.functional.batch_norm")
    assert fl[0] == x.numel() * 2  # with affine
    assert mc[0] == 0


def test_layer_norm_and_group_norm(patched_ops):
    ctx = patched_ops
    x = torch.randn(2, 4, 6, 8)
    _ = F.layer_norm(x, normalized_shape=(
        6, 8), weight=torch.ones(6, 8), bias=torch.zeros(6, 8))
    fl, mc = _take_records(ctx, "torch.nn.functional.layer_norm")
    assert fl[0] == x.numel() * 5
    assert mc[0] == 0

    _ = F.group_norm(x, num_groups=2, weight=torch.ones(4),
                     bias=torch.zeros(4))
    fl, mc = _take_records(ctx, "torch.nn.functional.group_norm")
    assert fl[0] == x.numel() * 5
    assert mc[0] == 0


def test_instance_norm(patched_ops):
    ctx = patched_ops
    x = torch.randn(2, 3, 7, 7)
    _ = F.instance_norm(x, running_mean=None,
                        running_var=None, use_input_stats=True)
    fl, mc = _take_records(ctx, "torch.nn.functional.instance_norm")
    assert fl[0] == x.numel() * 4  # no affine
    assert mc[0] == 0


# -----------------------------------------------------------------------------
# Einsum
# -----------------------------------------------------------------------------
def test_einsum_flops_positive(patched_ops):
    ctx = patched_ops
    i, j, k = 3, 4, 5
    a = torch.randn(i, j)
    b = torch.randn(j, k)
    _ = torch.einsum("ij,jk->ik", a, b)
    fl, mc = _take_records(ctx, "torch.einsum")
    assert isinstance(fl[0], int) and fl[0] > 0
    assert mc[0] == 0


# -----------------------------------------------------------------------------
# RNN hooks
# -----------------------------------------------------------------------------
def test_rnn_forward_hook_counts_and_bidir_double():
    lstm_uni = nn.LSTM(input_size=8, hidden_size=16,
                       num_layers=1, batch_first=True, bidirectional=False)
    lstm_bi = nn.LSTM(input_size=8, hidden_size=16,
                      num_layers=1, batch_first=True, bidirectional=True)

    h1 = lstm_uni.register_forward_hook(ops.MODULE_HOOK_MAPPING[nn.LSTM])
    h2 = lstm_bi.register_forward_hook(ops.MODULE_HOOK_MAPPING[nn.LSTM])

    x = torch.randn(2, 7, 8)
    lstm_uni.__flops__ = 0
    lstm_bi.__flops__ = 0

    _ = lstm_uni(x)
    _ = lstm_bi(x)

    h1.remove()
    h2.remove()

    assert hasattr(lstm_uni, "__flops__") and lstm_uni.__flops__ > 0
    assert hasattr(lstm_bi, "__flops__") and lstm_bi.__flops__ > 0
    assert lstm_bi.__flops__ == 2 * lstm_uni.__flops__


def test_rnn_cell_forward_hook_counts():
    cell = nn.LSTMCell(input_size=5, hidden_size=4)
    h = cell.register_forward_hook(ops.MODULE_HOOK_MAPPING[nn.LSTMCell])

    x = torch.randn(3, 5)
    h0 = torch.zeros(3, 4)
    c0 = torch.zeros(3, 4)
    cell.__flops__ = 0
    _ = cell(x, (h0, c0))
    h.remove()

    assert hasattr(cell, "__flops__")
    assert cell.__flops__ > 0
