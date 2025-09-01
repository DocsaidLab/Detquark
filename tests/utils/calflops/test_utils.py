from __future__ import annotations

import importlib
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

utils = importlib.import_module("utils.calflops.utils")


class FakeTokenizer:
    """A minimal tokenizer stub that mimics HF call signature/outputs.

    Args:
        support_token_type_ids: if True, returns token_type_ids.
        support_position_ids: if True, returns position_ids.
        pad_id: pad token id.
    """

    def __init__(self, support_token_type_ids: bool = True, support_position_ids: bool = False, pad_id: int = 0):
        self.support_token_type_ids = bool(support_token_type_ids)
        self.support_position_ids = bool(support_position_ids)
        self.pad_token_id = pad_id

    def __call__(
        self,
        texts: List[str],
        padding: str,
        max_length: int,
        truncation: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        return_tensors: str,
    ) -> Dict[str, torch.Tensor]:
        assert padding == "max_length"
        assert return_tensors == "pt"
        batch = len(texts)

        # Build [batch, max_length] tensors
        input_ids = torch.full((batch, max_length),
                               self.pad_token_id, dtype=torch.long)
        # put simple "specials"
        if max_length >= 2:
            input_ids[:, 0] = 1  # [CLS] alike
            input_ids[:, -1] = 2  # [SEP] alike

        out = {"input_ids": input_ids}

        if return_attention_mask:
            # here we mimic fully padded to length => attention all ones
            out["attention_mask"] = torch.ones(
                (batch, max_length), dtype=torch.long)

        if return_token_type_ids and self.support_token_type_ids:
            out["token_type_ids"] = torch.zeros(
                (batch, max_length), dtype=torch.long)

        if self.support_position_ids:
            pos = torch.arange(max_length, dtype=torch.long).unsqueeze(
                0).repeat(batch, 1)
            out["position_ids"] = pos

        return out


# ---------------------------------------------------------------------------
# generate_transformer_input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch,seq_len", [(1, 8), (4, 16)])
def test_generate_transformer_input_basic_shapes_cpu(batch, seq_len):
    tok = FakeTokenizer(support_token_type_ids=True,
                        support_position_ids=False)
    out = utils.generate_transformer_input(
        tokenizer=tok,
        input_shape=(batch, seq_len),
        device="cpu",
        add_position_ids=False,
    )
    assert set(out.keys()).issuperset({"input_ids", "attention_mask"})
    assert out["input_ids"].shape == (batch, seq_len)
    assert out["attention_mask"].shape == (batch, seq_len)
    # token_type_ids should exist for this tokenizer
    assert "token_type_ids" in out
    assert out["token_type_ids"].shape == (batch, seq_len)
    # dtype checks
    assert out["input_ids"].dtype == torch.long


def test_generate_transformer_input_no_token_type_ids_added_when_not_supported():
    tok = FakeTokenizer(support_token_type_ids=False,
                        support_position_ids=False)
    out = utils.generate_transformer_input(
        tok, input_shape=(2, 12), device="cpu", add_position_ids=False)
    assert "token_type_ids" not in out  # tokenizer didn't supply it


def test_generate_transformer_input_add_position_ids_when_missing():
    tok = FakeTokenizer(support_token_type_ids=True,
                        support_position_ids=False)
    batch, seq_len = 3, 10
    out = utils.generate_transformer_input(tok, input_shape=(
        batch, seq_len), device="cpu", add_position_ids=True)
    assert "position_ids" in out
    assert out["position_ids"].shape == (batch, seq_len)
    # Should be [0..seq_len-1]
    assert torch.all(out["position_ids"][0] == torch.arange(seq_len))


def test_generate_transformer_input_respects_existing_position_ids():
    tok = FakeTokenizer(support_token_type_ids=True, support_position_ids=True)
    out = utils.generate_transformer_input(
        tok, input_shape=(2, 7), device="cpu", add_position_ids=False)
    assert "position_ids" in out  # directly from tokenizer
    assert out["position_ids"].shape == (2, 7)


@pytest.mark.parametrize("bad_shape", [(0, 10), (2, 0), (1,), (1, 2, 3), "foo", None])
def test_generate_transformer_input_rejects_invalid_shapes(bad_shape):
    tok = FakeTokenizer()
    if bad_shape is None:
        # None is allowed (defaults to (1, 128)) – handled internally
        out = utils.generate_transformer_input(
            tok, input_shape=None, device="cpu", add_position_ids=False)
        assert out["input_ids"].shape == (1, 128)
    else:
        with pytest.raises(ValueError):
            utils.generate_transformer_input(
                tok, input_shape=bad_shape, device="cpu", add_position_ids=False)


# ---------------------------------------------------------------------------
# number/units formatting helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num,expected_prefix",
    [
        (1.2e12, "1.2 T"),
        (3.4e9, "3.4 G"),
        (5.6e6, "5.6 M"),
        (7.8e3, "7.8 K"),
        (999.0, "999"),
        (0.0, "0"),
        (1.2e-3, "1.2 m"),
        (9.9e-6, "9.9 u"),
    ],
)
def test_number_to_string_auto_units(num, expected_prefix):
    s = utils.number_to_string(num, precision=1)
    assert s.startswith(expected_prefix)


def test_number_to_string_negative_and_explicit_units():
    s1 = utils.number_to_string(-0.0012, precision=1)  # auto -> m
    assert s1 == "-1.2 m"
    s2 = utils.number_to_string(1_500_000, units="K", precision=1)
    assert s2 == "1500 K"
    # unknown unit -> magnitude=1
    s3 = utils.number_to_string(42, units="X", precision=2)
    assert s3 == "42 X"


def test_macs_flops_bytes_params_to_string_formatting():
    assert utils.macs_to_string(
        1.234e9, units="G", precision=2) == "1.23 GMACs"
    assert utils.flops_to_string(
        4.56e12, units="T", precision=2) == "4.56 TFLOPs"
    assert utils.bytes_to_string(1.5e6, units="M", precision=1) == "1.5 MB"

    # params_to_string with 'B' => display billions as B, not G
    s_b = utils.params_to_string(1_500_000_000, units="B", precision=1)
    assert s_b == "1.5 B"
    # auto (no units) -> uses G
    s_g = utils.params_to_string(1_500_000_000, units=None, precision=1)
    assert s_g == "1.5 G"


# ---------------------------------------------------------------------------
# convert_bytes (binary, 1024 base)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "0 B"),
        (1, "1 B"),
        (1023, "1023 B"),
        (1024, "1 KB"),
        (1536, "1.5 KB"),
        (1024**2, "1 MB"),
        (1024**3 * 3, "3 GB"),
    ],
)
def test_convert_bytes(value, expected):
    assert utils.convert_bytes(value) == expected


def test_convert_bytes_negative_raises():
    with pytest.raises(ValueError):
        utils.convert_bytes(-1)


# ---------------------------------------------------------------------------
# get_module_flops / get_module_macs
# ---------------------------------------------------------------------------

class Leaf(nn.Module):
    def __init__(self, flops: float, macs: float, param_shape=(4,)):
        super().__init__()
        self.__flops__ = float(flops)
        self.__macs__ = float(macs)
        # make param with controllable sparsity
        self.w = nn.Parameter(torch.ones(*param_shape, dtype=torch.float32))

    def zero_some(self, idxs: List[int]):
        with torch.no_grad():
            for i in idxs:
                self.w[i] = 0.0


class NoAttr(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3))


class Parent(nn.Module):
    def __init__(self, children):
        super().__init__()
        self.__flops__ = 40.0
        self.__macs__ = 20.0
        self.bias = nn.Parameter(torch.tensor([1.0, 0.0]))  # sparsity 0.5
        # register children as submodules
        for i, c in enumerate(children):
            self.add_module(f"child_{i}", c)


def test_module_accounting_recursive_and_sparse_scaling():
    a = Leaf(flops=100.0, macs=50.0, param_shape=(4,))
    a.zero_some([0, 1])  # 2/4 non-zero => 0.5 sparse factor

    b = Leaf(flops=200.0, macs=100.0, param_shape=(5,))
    # keep all non-zero => factor 1.0

    c = NoAttr()  # contributes 0 to __flops__/__macs__ but present in tree

    root = Parent(children=[a, b, c])

    # is_sparse = False
    total_flops = utils.get_module_flops(root, is_sparse=False)
    total_macs = utils.get_module_macs(root, is_sparse=False)
    assert total_flops == pytest.approx(40 + 100 + 200, rel=1e-6)
    assert total_macs == pytest.approx(20 + 50 + 100, rel=1e-6)

    # is_sparse = True
    total_flops_s = utils.get_module_flops(root, is_sparse=True)
    total_macs_s = utils.get_module_macs(root, is_sparse=True)
    # root sparsity: 1/2 => 0.5; a: 0.5; b: 1.0; c: no attrs (0)
    assert total_flops_s == pytest.approx(
        40 * 0.5 + 100 * 0.5 + 200 * 1.0, rel=1e-6)
    assert total_macs_s == pytest.approx(
        20 * 0.5 + 50 * 0.5 + 100 * 1.0, rel=1e-6)


def test_module_accounting_handles_missing_attrs_and_no_params():
    # A module with neither attrs nor params must not crash.
    class Empty(nn.Module):
        def __init__(self):
            super().__init__()

    root = Empty()
    assert utils.get_module_flops(root) == 0.0
    assert utils.get_module_macs(root) == 0.0


# ---------------------------------------------------------------------------
# is_package_available & alias
# ---------------------------------------------------------------------------

def test_is_package_available_on_existing_package():
    # pytest should be installed when running tests
    assert utils.is_package_available("pytest") is True


def test_is_package_available_on_nonexistent_package():
    assert utils.is_package_available(
        "___pkg_does_not_exist_12345___") is False


def test_is_package_available_alias():
    assert utils._is_package_available is utils.is_package_available
