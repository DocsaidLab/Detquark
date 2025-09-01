from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type

import torch.nn as nn

from .pytorch_ops import (MODULE_HOOK_MAPPING, _patch_functionals,
                          _patch_tensor_methods, _reload_functionals,
                          _reload_tensor_methods)
from .utils import (flops_to_string, get_module_flops, get_module_macs,
                    macs_to_string, params_to_string)

DEFAULT_PRECISION: int = 2


class CalFlopsPipeline:
    """Pipeline for estimating FLOPs/MACs/Params of a PyTorch model.

    The pipeline monkey-patches `torch`/`torch.nn.functional` ops and registers
    hooks on modules so that per-module and total FLOPs/MACs can be aggregated
    during a forward pass.

    Args:
        model: A `torch.nn.Module` to profile.
        include_backPropagation: If True, the report will additionally include
            fwd+bwd values using `compute_bp_factor`.
        compute_bp_factor: Backprop-to-forward ratio (default 2.0).
        is_sparse: If True, parameter counting uses non-zero entries only
            (`Tensor.count_nonzero()`), otherwise uses `numel()`.

    Notes:
        - Start via `start_flops_calculate()`, run a forward pass, then call
            `print_*` or the getters. Finally call `end_flops_calculate()` or use
            a `with` context to auto-clean.
    """

    def __init__(
        self,
        model: nn.Module,
        include_backPropagation: bool = False,
        compute_bp_factor: float = 2.0,
        is_sparse: bool = False,
    ) -> None:
        self.model = model
        self.include_backPropagation = bool(include_backPropagation)
        self.compute_bp_factor = float(compute_bp_factor)
        self.is_sparse = bool(is_sparse)

        # Internal state
        self._started: bool = False
        self._funcs_patched: bool = False
        self._old_functions: Dict[str, Callable[..., object]] = {}
        self._flop_stack: List[List[Tuple[str, int]]] = []
        self._mac_stack: List[List[Tuple[str, int]]] = []

        # Cached totals (filled after printing/getting)
        self.flops: Optional[float] = None
        self.macs: Optional[float] = None
        self.params: Optional[int] = None

    # --------------------------------------------------------------------------
    # Context manager helpers
    # --------------------------------------------------------------------------
    def __enter__(self) -> "CalFlopsPipeline":
        self.start_flops_calculate()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end_flops_calculate()

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _detach_module_hooks_and_clear_ignore(self) -> None:
        """Remove any previously attached hooks and clear ignore marks."""
        def _remove_hooks(module: nn.Module) -> None:
            if hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__.remove()
                delattr(module, "__pre_hook_handle__")
            if hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__.remove()
                delattr(module, "__post_hook_handle__")
            if hasattr(module, "__flops_handle__"):
                module.__flops_handle__.remove()
                delattr(module, "__flops_handle__")
            if hasattr(module, "__calflops_ignored__"):
                delattr(module, "__calflops_ignored__")

        self.model.apply(_remove_hooks)

    # --------------------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------------------
    def start_flops_calculate(
        self, ignore_list: Optional[Sequence[Type[nn.Module]]] = None
    ) -> None:
        """Start the profiling pipeline and attach hooks/patches.

        This resets per-module attributes and monkey-patches the functional/tensor
        APIs so that FLOPs/MACs will be recorded during the next forward pass.

        Args:
          ignore_list: Iterable of module classes to be ignored when attaching
            hooks (e.g., `[nn.Dropout]`). Exact `type(module)` matches are used.
        """
        # Reset counters on every start (idempotent for repeated starts).
        self.reset_flops_calculate()

        # Patch functionals/tensor methods using this instance's stacks (only once).
        if not self._funcs_patched:
            _patch_functionals(self._old_functions,
                               self._flop_stack, self._mac_stack)
            _patch_tensor_methods(self._old_functions,
                                  self._flop_stack, self._mac_stack)
            self._funcs_patched = True

        # 若已經有 hooks（例如在 __enter__ 內先 start 過），這裡先拆掉再按 ignore 規則重掛。
        self._detach_module_hooks_and_clear_ignore()

        def register_module_hooks(
            module: nn.Module, ignore: Optional[Sequence[Type[nn.Module]]]
        ) -> None:
            """Attach hooks per module; if module is ignored, isolate & drop its FLOPs."""
            # Determine ignore state (explicit type or inherited from ancestor).
            is_type_ignored = bool(ignore and type(module) in ignore)
            is_mark_ignored = bool(
                getattr(module, "__calflops_ignored__", False))
            ignored = is_type_ignored or is_mark_ignored

            if is_type_ignored and not is_mark_ignored:
                # Mark whole subtree as ignored so children enter the ignore branch.
                def _mark_ignored(m: nn.Module) -> None:
                    setattr(m, "__calflops_ignored__", True)
                module.apply(_mark_ignored)
                ignored = True

            if ignored:
                # Ignore branch: push/pop stacks but drop contributions entirely.
                def _pre_ignore(_module: nn.Module, _input) -> None:
                    self._flop_stack.append([])
                    self._mac_stack.append([])

                def _post_ignore(_module: nn.Module, _input, _output) -> None:
                    if self._flop_stack:
                        self._flop_stack.pop()
                    if self._mac_stack:
                        self._mac_stack.pop()

                module.__pre_hook_handle__ = module.register_forward_pre_hook(
                    _pre_ignore)
                module.__post_hook_handle__ = module.register_forward_hook(
                    _post_ignore)
                return

            # Non-ignored: RNN-like modules with dedicated hooks.
            if type(module) in MODULE_HOOK_MAPPING:
                module.__flops_handle__ = module.register_forward_hook(
                    MODULE_HOOK_MAPPING[type(module)]
                )
                return

            # General module: aggregate functionals via stack for this module only.
            def _pre_hook(_module: nn.Module, _input) -> None:
                self._flop_stack.append([])
                self._mac_stack.append([])

            def _post_hook(_module: nn.Module, _input, _output) -> None:
                if self._flop_stack:
                    flops_items = self._flop_stack.pop()
                    _module.__flops__ += int(sum(v for _, v in flops_items))
                if self._mac_stack:
                    macs_items = self._mac_stack.pop()
                    _module.__macs__ += int(sum(v for _, v in macs_items))

            module.__pre_hook_handle__ = module.register_forward_pre_hook(
                _pre_hook)
            module.__post_hook_handle__ = module.register_forward_hook(
                _post_hook)

        self.model.apply(
            partial(register_module_hooks, ignore=tuple(
                ignore_list) if ignore_list else None)
        )
        self._started = True

    def stop_flops_calculate(self) -> None:
        """Stop the profiling pipeline and remove hooks/patches."""
        # Restore monkey-patched functionals/tensor methods.
        if self._funcs_patched:
            _reload_functionals(self._old_functions)
            _reload_tensor_methods(self._old_functions)
            self._funcs_patched = False

        # Detach all module hooks and clear ignore marks.
        self._detach_module_hooks_and_clear_ignore()

    def reset_flops_calculate(self) -> None:
        """Reset per-module counters and (own) parameter counts."""
        def _count_params(m: nn.Module) -> int:
            if self.is_sparse:
                return sum(
                    p.count_nonzero().item() for p in m.parameters(recurse=False) if p.requires_grad
                )
            return sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)

        def _add_or_reset_attrs(module: nn.Module) -> None:
            module.__flops__ = 0
            module.__macs__ = 0
            module.__params__ = _count_params(module)

        self.model.apply(_add_or_reset_attrs)

    def end_flops_calculate(self) -> None:
        """End the pipeline, restore originals, and remove all attributes."""
        if not self._started:
            return
        self.stop_flops_calculate()
        self._started = False

        def _cleanup(module: nn.Module) -> None:
            for attr in (
                "__flops__",
                "__macs__",
                "__params__",
                "original_extra_repr",
                "__calflops_ignored__",
            ):
                if hasattr(module, attr):
                    delattr(module, attr)

        self.model.apply(_cleanup)

    # --------------------------------------------------------------------------
    # Totals & formatting
    # --------------------------------------------------------------------------
    def _sum_model_params(self) -> int:
        """Sum all trainable parameters across the whole model (owns + children)."""
        if self.is_sparse:
            return sum(p.count_nonzero().item() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_total_flops(self, as_string: bool = False):
        """Return total FLOPs of the model forward pass."""
        total_flops = get_module_flops(self.model, is_sparse=self.is_sparse)
        return flops_to_string(total_flops) if as_string else total_flops

    def get_total_macs(self, as_string: bool = False):
        """Return total MACs of the model forward pass."""
        total_macs = get_module_macs(self.model, is_sparse=self.is_sparse)
        return macs_to_string(total_macs) if as_string else total_macs

    def get_total_params(self, as_string: bool = False):
        """Return total trainable parameters of the model (per rank)."""
        total_params = self._sum_model_params()
        return params_to_string(total_params) if as_string else total_params

    # --------------------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------------------
    def _make_report(
        self,
        units: Optional[str],
        precision: int,
        print_detailed: bool,
        print_results: bool,
    ) -> str:
        if not self._started:
            return ""

        total_flops = float(self.get_total_flops(as_string=False))
        total_macs = float(self.get_total_macs(as_string=False))
        total_params = int(self.get_total_params(as_string=False))

        # cache
        self.flops = total_flops
        self.macs = total_macs
        self.params = total_params

        lines: List[str] = []
        lines.append(
            "\n------------------------------------- Calculate Flops Results -------------------------------------"
        )
        lines.append(
            "Notations:\n"
            "number of parameters (Params), number of multiply-accumulate operations (MACs),\n"
            "number of floating-point operations (FLOPs), floating-point operations per second (FLOPS),\n"
            "fwd FLOPs (forward), bwd FLOPs (backward),\n"
            f"default model backpropagation takes {self.compute_bp_factor:.2f}× the forward computation.\n"
        )

        line_fmt = "{:<70}  {:<12}"
        lines.append(line_fmt.format("Total Training Params:",
                     params_to_string(total_params)))
        lines.append(
            line_fmt.format(
                "fwd MACs:", macs_to_string(
                    total_macs, units=units, precision=precision)
            )
        )
        lines.append(
            line_fmt.format(
                "fwd FLOPs:", flops_to_string(
                    total_flops, units=units, precision=precision)
            )
        )

        if self.include_backPropagation:
            lines.append(
                line_fmt.format(
                    "fwd+bwd MACs:",
                    macs_to_string(
                        total_macs * (1 + self.compute_bp_factor), units=units, precision=precision
                    ),
                )
            )
            lines.append(
                line_fmt.format(
                    "fwd+bwd FLOPs:",
                    flops_to_string(
                        total_flops * (1 + self.compute_bp_factor), units=units, precision=precision
                    ),
                )
            )

        def _flops_repr(module: nn.Module) -> str:
            """extra_repr replacement showing per-module params/MACs/FLOPs and percentages."""
            params = int(getattr(module, "__params__", 0))
            flops = float(get_module_flops(module, is_sparse=self.is_sparse))
            macs = float(get_module_macs(module, is_sparse=self.is_sparse))
            items = [
                f"{params_to_string(params)} = {round(100 * params / total_params, precision) if total_params else 0:g}% Params",
                f"{macs_to_string(macs)} = {round(100 * macs / total_macs, precision) if total_macs else 0:g}% MACs",
                f"{flops_to_string(flops)} = {round(100 * flops / total_flops, precision) if total_flops else 0:g}% FLOPs",
            ]
            orig = module.original_extra_repr() if hasattr(
                module, "original_extra_repr") else None
            if orig:
                items.append(orig)
            return ", ".join(items)

        def _add_extra_repr(module: nn.Module) -> None:
            # Bind and inject extra_repr for pretty printing.
            flops_extra_repr = _flops_repr.__get__(module)  # bind to instance
            if getattr(module, "extra_repr", None) is not flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                # type: ignore[assignment]
                module.extra_repr = flops_extra_repr

        def _del_extra_repr(module: nn.Module) -> None:
            if hasattr(module, "original_extra_repr"):
                # type: ignore[assignment]
                module.extra_repr = module.original_extra_repr
                delattr(module, "original_extra_repr")

        self.model.apply(_add_extra_repr)

        if print_detailed:
            lines.append(
                "\n-------------------------------- Detailed Calculated FLOPs Results --------------------------------"
            )
            lines.append(
                "Each module is listed as: params(%), MACs(%), FLOPs(%), followed by the original extra_repr.\n"
                "Note:\n"
                "1) Some logits/loss computations are functional-only and not submodules; they contribute to the\n"
                "   parent's totals but are not displayed as children here.\n"
                "2) FLOPs are theoretical estimates and can exceed system throughput.\n"
            )
            lines.append(str(self.model))

        self.model.apply(_del_extra_repr)

        lines.append(
            "---------------------------------------------------------------------------------------------------")

        report = "\n".join(lines)
        if print_results:
            print(report)
        return report

    # Backward-compatible method names (returning string)
    def print_return_model_pipeline(
        self,
        units: Optional[str] = None,
        precision: int = DEFAULT_PRECISION,
        print_detailed: bool = True,
        print_results: bool = True,
    ) -> str:
        """Return (and optionally print) the report string."""
        return self._make_report(
            units=units, precision=precision, print_detailed=print_detailed, print_results=print_results
        )

    # Backward-compatible alias (original mis-spelled name)
    def print_return_model_pipline(
        self,
        units: Optional[str] = None,
        precision: int = DEFAULT_PRECISION,
        print_detailed: bool = True,
        print_results: bool = True,
    ) -> str:
        return self.print_return_model_pipeline(
            units=units, precision=precision, print_detailed=print_detailed, print_results=print_results
        )

    # Printing-only variant (kept for API parity)
    def print_model_pipeline(
        self, units: Optional[str] = None, precision: int = DEFAULT_PRECISION, print_detailed: bool = True
    ) -> None:
        """Print the report to stdout."""
        self._make_report(units=units, precision=precision,
                          print_detailed=print_detailed, print_results=True)

    # Backward-compatible alias (original mis-spelled name)
    def print_model_pipline(
        self, units: Optional[str] = None, precision: int = DEFAULT_PRECISION, print_detailed: bool = True
    ) -> None:
        self.print_model_pipeline(
            units=units, precision=precision, print_detailed=print_detailed)


# ------------------------------------------------------------------------------
# Backward-compat class alias
# ------------------------------------------------------------------------------
CalFlopsPipline = CalFlopsPipeline
