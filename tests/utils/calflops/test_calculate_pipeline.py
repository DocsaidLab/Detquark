from __future__ import annotations

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

pl_mod = importlib.import_module("utils.calflops.calculate_pipeline")
u_mod = importlib.import_module("utils.calflops.utils")
CalFlopsPipeline = getattr(pl_mod, "CalFlopsPipeline", None)


# -----------------------------------------------------------------------------
# Helper to build a small, deterministic model
# -----------------------------------------------------------------------------
def _make_toy_model():
    # Conv -> ReLU -> AdaptiveAvgPool2d(8x8) -> Flatten -> Linear(256->10)
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=False),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 10, bias=True),
    )


# -----------------------------------------------------------------------------
# Idempotent patch/restore via pipeline lifecycle
# -----------------------------------------------------------------------------
def test_pipeline_patch_and_restore_idempotent():
    model = nn.Sequential(nn.ReLU())
    pipe = CalFlopsPipeline(model)

    # Record original function ids
    orig_ids = {
        "linear": id(F.linear),
        "conv2d": id(F.conv2d),
        "interpolate": id(F.interpolate),
        "matmul": id(torch.matmul),
        "addmm": id(torch.addmm),
    }

    pipe.start_flops_calculate()
    try:
        # After start, functions must be wrapped
        assert id(F.linear) != orig_ids["linear"]
        assert id(F.conv2d) != orig_ids["conv2d"]
        assert id(F.interpolate) != orig_ids["interpolate"]
        assert id(torch.matmul) != orig_ids["matmul"]
        assert id(torch.addmm) != orig_ids["addmm"]
    finally:
        pipe.end_flops_calculate()

    # After end, functions restored
    assert id(F.linear) == orig_ids["linear"]
    assert id(F.conv2d) == orig_ids["conv2d"]
    assert id(F.interpolate) == orig_ids["interpolate"]
    assert id(torch.matmul) == orig_ids["matmul"]
    assert id(torch.addmm) == orig_ids["addmm"]


# -----------------------------------------------------------------------------
# Baseline totals (MACs/FLOPs/Params) for a simple CNN+MLP
# -----------------------------------------------------------------------------
def _expected_baseline_numbers():
    # Input: N=2, C=3, H=W=8
    N, Cin, H, W = 2, 3, 8, 8
    # Conv: outC=4, k=3, padding=1 -> output (N, 4, 8, 8)
    Cout, k = 4, 3
    elements = N * H * W  # per-output-position count across batch
    conv_macs = elements * (Cin * (k * k) * Cout)  # 128 * (3*9*4) = 13824
    conv_flops = 2 * conv_macs + (elements * Cout)  # + bias adds

    # ReLU FLOPs = numel of conv output
    relu_flops = N * Cout * H * W  # 2*4*8*8 = 512

    # AdaptiveAvgPool2d(8x8): FLOPs approx = num output elements
    pool_flops = N * Cout * H * W  # 512

    # Linear: in=4*8*8=256, out=10, batch=2
    in_features = 256
    out_features = 10
    num_instances = N
    lin_macs = num_instances * in_features * out_features  # 2*256*10 = 5120
    lin_flops = 2 * lin_macs + \
        (num_instances * out_features)  # + bias adds = 10260

    total_macs = conv_macs + lin_macs  # 18944
    total_flops = conv_flops + relu_flops + pool_flops + lin_flops  # 39444

    # Params: conv(4*3*3*3 + 4) + linear(10*256 + 10) = 108+4 + 2560+10 = 2682
    total_params = (Cout * Cin * k * k + Cout) + \
        (out_features * in_features + out_features)

    return total_macs, total_flops, total_params, relu_flops


def test_pipeline_basics_and_totals():
    model = _make_toy_model()
    x = torch.randn(2, 3, 8, 8)

    exp_macs, exp_flops, exp_params, _ = _expected_baseline_numbers()

    with CalFlopsPipeline(model) as pipe:
        _ = model(x)
        total_macs = pipe.get_total_macs(as_string=False)
        total_flops = pipe.get_total_flops(as_string=False)
        total_params = pipe.get_total_params(as_string=False)

        # Totals match the expected hand calculation
        assert total_macs == exp_macs
        assert total_flops == exp_flops
        assert total_params == exp_params

        # Own-only params on Conv layer (recurse=False) = 4*3*3*3 + 4
        conv0: nn.Conv2d = model[0]  # type: ignore[assignment]
        assert hasattr(conv0, "__params__")
        assert conv0.__params__ == (4 * 3 * 3 * 3 + 4)

    # After context exit, attributes are cleaned up
    assert not hasattr(model[0], "__flops__")
    assert not hasattr(model[0], "__macs__")
    assert not hasattr(model[0], "__params__")


# -----------------------------------------------------------------------------
# Ignore list: skip counting ReLU module
# -----------------------------------------------------------------------------
def test_pipeline_ignore_list_skips_module():
    model = _make_toy_model()
    x = torch.randn(2, 3, 8, 8)

    exp_macs, exp_flops, _, relu_flops = _expected_baseline_numbers()

    # Baseline
    with CalFlopsPipeline(model) as pipe0:
        _ = model(x)
        baseline_flops = pipe0.get_total_flops(as_string=False)
        baseline_macs = pipe0.get_total_macs(as_string=False)
        assert baseline_flops == exp_flops
        assert baseline_macs == exp_macs

    # Ignore ReLU
    with CalFlopsPipeline(model) as pipe1:
        # explicit start with ignore list
        pipe1.start_flops_calculate(ignore_list=[nn.ReLU])
        _ = model(x)
        flops_ignored = pipe1.get_total_flops(as_string=False)
        macs_ignored = pipe1.get_total_macs(as_string=False)

    # ReLU contributes only FLOPs (no MACs). Difference should be exactly ReLU FLOPs.
    assert baseline_macs == macs_ignored
    assert baseline_flops - flops_ignored == relu_flops


# -----------------------------------------------------------------------------
# Report string contents & bwd factor
# -----------------------------------------------------------------------------
def test_pipeline_report_contains_expected_strings(capsys):
    model = _make_toy_model()
    x = torch.randn(2, 3, 8, 8)
    exp_macs, exp_flops, exp_params, _ = _expected_baseline_numbers()

    with CalFlopsPipeline(model, include_backPropagation=True, compute_bp_factor=2.0) as pipe:
        _ = model(x)
        report = pipe.print_return_model_pipeline(
            units=None, precision=2, print_detailed=True, print_results=False
        )

        # Lines must include totals (use utils' formatters to avoid brittle formatting)
        fwd_macs_s = u_mod.macs_to_string(exp_macs, units=None, precision=2)
        fwd_flops_s = u_mod.flops_to_string(exp_flops, units=None, precision=2)
        total_params_s = u_mod.params_to_string(exp_params)

        fwd_bwd_macs_s = u_mod.macs_to_string(
            exp_macs * 3, units=None, precision=2)  # 1 + 2.0 = 3
        fwd_bwd_flops_s = u_mod.flops_to_string(
            exp_flops * 3, units=None, precision=2)

        assert "Calculate Flops Results" in report
        assert fwd_macs_s in report
        assert fwd_flops_s in report
        assert total_params_s in report
        assert fwd_bwd_macs_s in report
        assert fwd_bwd_flops_s in report

        # Ensure we cleaned up extra_repr wrappers
        for m in model.modules():
            assert not hasattr(m, "original_extra_repr")

    # print-only variant (smoke test)
    with CalFlopsPipeline(model) as pipe2:
        _ = model(x)
        pipe2.print_model_pipeline(
            units=None, precision=2, print_detailed=False)
    out = capsys.readouterr().out
    assert "Calculate Flops Results" in out


# -----------------------------------------------------------------------------
# Sparse parameter counting (count_nonzero)
# -----------------------------------------------------------------------------
def test_pipeline_sparse_params_counting():
    class ToyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 4, bias=True)

        def forward(self, x):
            return self.fc(x)

    net = ToyNet()
    # Make half of weights zero; bias all zeros
    with torch.no_grad():
        w = net.fc.weight  # (4,6) => 24 params
        w[:] = 1.0
        w.view(-1)[:12] = 0.0  # zero out half
        net.fc.bias[:] = 0.0   # 4 zeros

    dense_total = 24 + 4
    sparse_total = 12 + 0

    # Dense counting
    p_dense = CalFlopsPipeline(net, is_sparse=False)
    assert p_dense.get_total_params(as_string=False) == dense_total

    # Sparse counting
    p_sparse = CalFlopsPipeline(net, is_sparse=True)
    assert p_sparse.get_total_params(as_string=False) == sparse_total


# -----------------------------------------------------------------------------
# RNN hooks (bidirectional is exactly 2x)
# -----------------------------------------------------------------------------
def test_pipeline_rnn_bidirectional_double():
    # Unidirectional
    lstm_uni = nn.LSTM(input_size=8, hidden_size=16,
                       num_layers=1, batch_first=True, bidirectional=False)
    x = torch.randn(2, 7, 8)
    with CalFlopsPipeline(lstm_uni) as p1:
        _ = lstm_uni(x)
        flops_uni = p1.get_total_flops(as_string=False)

    # Bidirectional
    lstm_bi = nn.LSTM(input_size=8, hidden_size=16,
                      num_layers=1, batch_first=True, bidirectional=True)
    with CalFlopsPipeline(lstm_bi) as p2:
        _ = lstm_bi(x)
        flops_bi = p2.get_total_flops(as_string=False)

    assert flops_uni > 0
    assert flops_bi == 2 * flops_uni


# -----------------------------------------------------------------------------
# Re-entrant start (no double hooks) and stability
# -----------------------------------------------------------------------------
def test_pipeline_multiple_starts_are_safe():
    model = _make_toy_model()
    pipe = CalFlopsPipeline(model)

    # Calling start twice should not crash or duplicate hooks
    pipe.start_flops_calculate()
    pipe.start_flops_calculate()
    x = torch.randn(2, 3, 8, 8)
    _ = model(x)
    # Query totals (smoke)
    _ = pipe.get_total_flops()
    _ = pipe.get_total_macs()
    _ = pipe.get_total_params()

    # End must restore successfully
    pipe.end_flops_calculate()

    # No lingering attributes
    for m in model.modules():
        assert not hasattr(m, "__pre_hook_handle__")
        assert not hasattr(m, "__post_hook_handle__")
        assert not hasattr(m, "__flops_handle__")
