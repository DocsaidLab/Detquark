from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from .calculate_pipeline import CalFlopsPipeline as _Pipeline
from .utils import (flops_to_string, generate_transformer_input,
                    macs_to_string, params_to_string)


def _to_device(obj: Any, device: torch.device) -> Any:
    """Move a tensor (or nested containers of tensors) to the target device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def calculate_flops(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    transformer_tokenizer: Optional[Any] = None,
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    forward_mode: str = "forward",
    include_backPropagation: bool = False,
    compute_bp_factor: float = 2.0,
    print_results: bool = True,
    print_detailed: bool = True,
    output_as_string: bool = True,
    output_precision: int = 2,
    output_unit: Optional[str] = None,
    ignore_modules: Optional[Sequence[Type[nn.Module]]] = None,
    is_sparse: bool = False,
) -> Tuple[Union[str, float], Union[str, float], Union[str, int]]:
    """Profile a model and return total FLOPs, MACs, and Params.

    The function uses an internal pipeline to monkey‑patch relevant PyTorch ops,
    runs one forward (or `generate`) pass, aggregates FLOPs/MACs/Params and then
    restores the original functions and hooks.

    Args:
        model: A PyTorch module to be profiled (will be run in `.eval()` mode).
        input_shape: If provided and `args`/`kwargs` are empty, a random tensor
            with this shape is generated and passed as the *only* positional arg.
            Format for transformers models: `(batch_size, seq_len)`.
        transformer_tokenizer: Hugging Face tokenizer. If provided and you don't
            pass `args/kwargs`, a proper transformer input dict is auto-generated.
        args: Positional args to call the model with (optional).
        kwargs: Keyword args to call the model with (optional).
        forward_mode: `"forward"` to call `model(*args, **kwargs)`, `"generate"`
            to call `model.generate(*args, **kwargs)`.
        include_backPropagation: If True, final returned numbers include
            backward cost via `compute_bp_factor` (printing also shows fwd+bwd).
        compute_bp_factor: Backward-to-forward compute ratio (default 2.0).
        print_results: Whether to print the human-readable report.
        print_detailed: Whether to include the per-module breakdown in the print.
        output_as_string: If True, return values are human-readable strings with
            SI units; otherwise raw numeric totals are returned.
        output_precision: Decimal places for string output.
        output_unit: Force output unit (e.g., `"T"`, `"G"`, `"M"`, `"K"`). If
            None, the unit is chosen automatically based on magnitude.
        ignore_modules: Iterable of module classes to ignore (e.g., `[nn.ReLU]`).
            The entire subtree rooted at instances of these classes will be
            excluded from FLOPs/MACs accounting.
        is_sparse: If True, parameter counting uses `count_nonzero()`, else
            uses full `numel()`.

    Returns:
        A tuple `(flops, macs, params)` where each item is either a string (if
        `output_as_string=True`) or a numeric value (float/int).

    Raises:
        AssertionError: If inputs are inconsistent (e.g., `input_shape` provided
            together with non-empty `args/kwargs`).
        NotImplementedError: If `forward_mode` is not supported.
    """
    assert isinstance(model, nn.Module), "model must be a PyTorch nn.Module"
    model.eval()

    # Prepare argument containers (avoid mutable defaults).
    args = list(args) if args is not None else []
    kwargs = dict(kwargs) if kwargs is not None else {}

    # Determine device/dtype safely (models without parameters fallback to CPU/float32).
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device(
        "cpu")
    dtype = first_param.dtype if first_param is not None else torch.float32
    model.to(device)

    # Heuristic: treat as transformers when tokenizer is given or class path contains "transformers".
    is_transformer = bool(
        transformer_tokenizer is not None
        or "transformers" in getattr(model.__class__, "__module__", "")
    )

    # -----------------------------------------------------------------------
    # Build inputs if not explicitly provided
    # -----------------------------------------------------------------------
    if input_shape is not None:
        # If input_shape is specified, the call expects auto-generated inputs.
        if len(args) > 0 or len(kwargs) > 0:
            raise AssertionError(
                "When input_shape is given, args and kwargs must be empty; "
                "inputs will be generated automatically."
            )
        if not isinstance(input_shape, tuple) or len(input_shape) < 1:
            raise AssertionError("input_shape must be a non-empty tuple.")

        if transformer_tokenizer is None:
            # For non-transformers, feed a raw tensor with given shape.
            if is_transformer:
                raise AssertionError(
                    "Transformer model requires a tokenizer to auto-generate inputs "
                    "when input_shape is provided."
                )
            dummy_input = torch.empty(input_shape, dtype=dtype, device=device)
            args = [dummy_input]
        else:
            # For transformers, enforce (batch_size, seq_len) and build kwargs.
            if len(input_shape) != 2:
                raise AssertionError(
                    "For transformers, input_shape must be (batch_size, seq_len)."
                )
            kwargs = generate_transformer_input(
                tokenizer=transformer_tokenizer,
                input_shape=input_shape,
                device=device,
            )
    else:
        # No input_shape: either user provided args/kwargs, or we can still
        # auto-generate transformers inputs if a tokenizer is provided.
        if transformer_tokenizer is not None and len(args) == 0 and len(kwargs) == 0:
            kwargs = generate_transformer_input(
                tokenizer=transformer_tokenizer, input_shape=None, device=device
            )
        if len(args) == 0 and len(kwargs) == 0:
            raise AssertionError(
                "You must provide either input_shape, or args/kwargs, or a tokenizer to auto-generate inputs."
            )

    # Move all inputs to the model's device (tensors only).
    args = [_to_device(a, device) for a in args]
    kwargs = _to_device(kwargs, device) if kwargs else {}

    # -----------------------------------------------------------------------
    # Run the profiling pipeline
    # -----------------------------------------------------------------------
    pipeline = _Pipeline(
        model=model,
        include_backPropagation=include_backPropagation,
        compute_bp_factor=compute_bp_factor,
        is_sparse=is_sparse,
    )

    try:
        pipeline.start_flops_calculate(ignore_list=ignore_modules)

        if forward_mode == "forward":
            _ = model(*args, **kwargs)
        elif forward_mode == "generate":
            if not hasattr(model, "generate"):
                raise NotImplementedError(
                    "forward_mode='generate' requires the model to define `.generate()`."
                )
            _ = model.generate(*args, **kwargs)
        else:
            raise NotImplementedError(
                "forward_mode should be either 'forward' or 'generate'.")

        # Forward-only totals from the pipeline
        flops = float(pipeline.get_total_flops(as_string=False))
        macs = float(pipeline.get_total_macs(as_string=False))
        params = int(pipeline.get_total_params(as_string=False))

        # Optional printing (pipeline handles detailed formatting)
        if print_results:
            pipeline.print_model_pipeline(
                units=output_unit, precision=output_precision, print_detailed=print_detailed
            )

    finally:
        # Ensure hooks and patches are removed even if forward raises.
        pipeline.end_flops_calculate()

    # Optionally include backward cost in the returned values.
    if include_backPropagation:
        factor = 1.0 + float(compute_bp_factor)
        flops *= factor
        macs *= factor

    if output_as_string:
        return (
            flops_to_string(flops, units=output_unit,
                            precision=output_precision),
            macs_to_string(macs, units=output_unit,
                           precision=output_precision),
            params_to_string(params, units=output_unit,
                             precision=output_precision),
        )

    return flops, macs, params
