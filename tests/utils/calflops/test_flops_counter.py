from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

import pytest
import torch
import torch.nn as nn

fc_mod = importlib.import_module("utils.calflops.flops_counter")
utils_mod = importlib.import_module("utils.calflops.utils")
calculate_flops = fc_mod.calculate_flops


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_toy_cnn() -> nn.Module:
    # Conv(3->4,k3,p1) -> ReLU -> AdaptiveAvgPool(8x8) -> Flatten -> Linear(256->10)
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=False),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 10, bias=True),
    )


def expected_cnn_macs_flops_params(
    batch: int, cin: int = 3, cout: int = 4, h: int = 8, w: int = 8, k: int = 3
) -> Tuple[int, int, int]:
    """Return (macs, flops_low, params).
    flops_low 不含 Linear bias-加法（視 PyTorch 內核是否顯式計算加法而定）。
    """
    # Conv2d
    elements = batch * h * w
    conv_macs = elements * (cin * (k * k) * cout)
    conv_flops = 2 * conv_macs + (elements * cout)  # bias adds

    # ReLU
    relu_flops = batch * cout * h * w

    # AdaptiveAvgPool2d(8,8)
    pool_flops = batch * cout * 8 * 8

    # Linear (256->10) with batch
    in_features = cout * 8 * 8
    out_features = 10
    lin_macs = batch * in_features * out_features
    lin_flops_no_bias = 2 * lin_macs  # bias 加法是否計入視算子實作；我們做寬鬆比對

    macs = conv_macs + lin_macs
    flops_low = conv_flops + relu_flops + pool_flops + lin_flops_no_bias

    # Params
    params = (cout * cin * k * k + cout) + \
        (out_features * in_features + out_features)
    return macs, flops_low, params


class TinyGenerateNet(nn.Module):
    """A tiny module that supports .generate(), for forward_mode='generate' path."""

    def __init__(self, in_f: int = 6, out_f: int = 5, bias: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    # generate 與 forward 同簽名，直接呼叫 forward
    # type: ignore[override]
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class StubTokenizer:
    """Minimal HF-like tokenizer stub for generate_transformer_input().

    修正點：實作 __call__ 以符合 utils.generate_transformer_input 的介面。
    """

    pad_token_id = 0

    def __call__(
        self,
        texts,
        *,
        padding: str = "max_length",
        max_length: int = 16,
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
        return_tensors: str = "pt",
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(texts)
        # 構造最小可用 batch：全 1 的 input_ids、全 1 的 attention_mask
        input_ids = torch.ones((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # 視需求附帶 token_type_ids（forward 端若不吃，模型需能吞 **kwargs）
        if return_token_type_ids:
            out["token_type_ids"] = torch.zeros(
                (batch_size, max_length), dtype=torch.long)

        # return_tensors="pt" 已由上方回傳 torch.Tensor 滿足
        return out


class TinyTransformerLike(nn.Module):
    """A minimal 'transformer-like' module that consumes (input_ids, attention_mask).

    修正點：forward 接受 **kwargs，忽略多餘鍵（例如 token_type_ids）。
    """

    def __init__(self, vocab: int = 200, dim: int = 16, out_f: int = 8, bias: bool = False) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        # bias=False 以避免對 Linear bias FLOPs 的依賴
        self.proj = nn.Linear(dim, out_f, bias=bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs: Any,  # 忽略 token_type_ids 等多餘項
    ) -> torch.Tensor:
        x = self.emb(input_ids)              # (B, S, D)
        y = self.proj(x)                     # (B, S, O)
        return y.sum(dim=1)                  # (B, O)，sum 的 FLOPs 未計入（未 patch）


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_cnn_numeric_and_string_and_print(capsys):
    model = make_toy_cnn()
    B = 2
    macs_exp, flops_low, params_exp = expected_cnn_macs_flops_params(B)

    # 數值輸出
    flops_val, macs_val, params_val = calculate_flops(
        model=model,
        input_shape=(B, 3, 8, 8),
        print_results=False,
        print_detailed=True,
        output_as_string=False,
    )
    # MACs 與 Params 為確定值
    assert macs_val == macs_exp
    assert params_val == params_exp
    # FLOPs 可能包含 Linear bias 加法（+ B * out_features = +20）
    assert flops_val in (flops_low, flops_low + B * 10)

    # 字串輸出 + 列印檢查（用工具函式產生期望字串，避免對大小寫/單位硬編碼）
    flops_str, macs_str, params_str = calculate_flops(
        model=model,
        input_shape=(B, 3, 8, 8),
        print_results=True,
        print_detailed=False,
        output_as_string=True,
        output_precision=2,
        output_unit=None,
    )
    out = capsys.readouterr().out
    assert "Calculate Flops Results" in out
    # 直接對照回傳字串是否出現在列印內容中
    assert flops_str in out
    assert macs_str in out
    assert params_str in out


def test_include_backprop_scales_results():
    model = make_toy_cnn()
    B = 2
    # 先取 baseline（FWD）
    flops_fwd, macs_fwd, params_val = calculate_flops(
        model=model, input_shape=(B, 3, 8, 8), print_results=False, output_as_string=False
    )
    # 啟用回傳的反向倍數（不影響 Params）
    flops_all, macs_all, params_all = calculate_flops(
        model=model,
        input_shape=(B, 3, 8, 8),
        include_backPropagation=True,
        compute_bp_factor=2.0,
        print_results=False,
        output_as_string=False,
    )
    assert params_all == params_val
    assert pytest.approx(flops_all, rel=0, abs=1e-6) == flops_fwd * 3.0
    assert pytest.approx(macs_all, rel=0, abs=1e-6) == macs_fwd * 3.0


def test_ignore_relu_excludes_flops_not_macs():
    model = make_toy_cnn()
    B, C, H, W = 2, 4, 8, 8
    # baseline
    flops0, macs0, _ = calculate_flops(
        model=model, input_shape=(B, 3, H, W), print_results=False, output_as_string=False
    )
    # 忽略 ReLU 子樹
    flops1, macs1, _ = calculate_flops(
        model=model,
        input_shape=(B, 3, H, W),
        ignore_modules=[nn.ReLU],
        print_results=False,
        output_as_string=False,
    )
    # ReLU 的 FLOPs = numel(conv_out) = B*C*H*W
    relu_flops = B * C * H * W
    assert macs0 == macs1
    assert flops0 - flops1 == relu_flops


def test_forward_mode_generate_path_smoke():
    model = TinyGenerateNet(in_f=6, out_f=5, bias=False)
    x = torch.randn(4, 6)
    # 不要求精確 FLOPs（避免與 Linear bias 細節耦合），但至少 > 0 且不報錯
    flops_v, macs_v, params_v = calculate_flops(
        model=model,
        args=[x],
        print_results=False,
        output_as_string=False,
        forward_mode="generate",
    )
    assert flops_v > 0
    assert macs_v > 0
    assert params_v == sum(p.numel()
                           for p in model.parameters() if p.requires_grad)


def test_transformer_autoinput_with_tokenizer_stub_numeric():
    tokenizer = StubTokenizer()
    model = TinyTransformerLike(vocab=200, dim=16, out_f=8, bias=False)

    B, S = 2, 10
    flops_v, macs_v, params_v = calculate_flops(
        model=model,
        input_shape=(B, S),
        transformer_tokenizer=tokenizer,
        print_results=False,
        output_as_string=False,
    )
    # 預期僅 Linear 產生 FLOPs/MACs： (B*S*D*O) MACs, FLOPs=2*MACs
    D, O = 16, 8
    macs_exp = B * S * D * O
    flops_exp = 2 * macs_exp  # bias=False
    assert macs_v == macs_exp
    assert flops_v == flops_exp
    # 參數數量（Embedding + Linear）
    params_exp = model.emb.num_embeddings * model.emb.embedding_dim + (D * O)
    assert params_v == params_exp


def test_invalid_forward_mode_and_input_conflicts():
    model = make_toy_cnn()
    # forward_mode 無效
    with pytest.raises(NotImplementedError):
        _ = calculate_flops(
            model=model, input_shape=(1, 3, 8, 8), print_results=False, forward_mode="unknown"
        )
    # input_shape 與 args/kwargs 衝突
    with pytest.raises(AssertionError):
        _ = calculate_flops(
            model=model, input_shape=(1, 3, 8, 8), args=[torch.randn(1, 3, 8, 8)], print_results=False
        )
    with pytest.raises(AssertionError):
        _ = calculate_flops(
            model=model, input_shape=(1, 3, 8, 8), kwargs={"x": torch.randn(1, 3, 8, 8)}, print_results=False
        )


def test_sparse_params_counting():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 4, bias=True)

        def forward(self, x):
            return self.fc(x)

    net = Net()
    # 讓一半權重為 0，bias 全 0
    with torch.no_grad():
        w = net.fc.weight  # (4,6) => 24
        w[:] = 1.0
        w.view(-1)[:12] = 0.0
        net.fc.bias[:] = 0.0  # 4

    # dense params: 24 + 4 = 28；sparse params: 12 + 0 = 12
    x = torch.randn(2, 6)
    _, _, dense_params = calculate_flops(
        model=net, args=[x], print_results=False, output_as_string=False, is_sparse=False
    )
    _, _, sparse_params = calculate_flops(
        model=net, args=[x], print_results=False, output_as_string=False, is_sparse=True
    )
    assert dense_params == 28
    assert sparse_params == 12
