import sys
from typing import Any, Iterable, List, Optional

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm


class CustomTQDMProgressBar(TQDMProgressBar):
    """
    Custom TQDM progress bar with adjustable unit scale and unified appearance.
    Extends Lightning's TQDMProgressBar to support a scaling factor for progress units
    and consistent bar configuration across training, validation, test, and predict.
    """

    def __init__(
        self,
        unit_scale: float | bool = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            unit_scale: Factor to scale the progress "unit" (e.g., samples per iteration).
            *args, **kwargs: Passed to base TQDMProgressBar initializer.
        """
        super().__init__(*args, **kwargs)
        self.unit_scale = unit_scale

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _resolve_total(total: Any) -> Optional[int]:
        """
        1. 如果是 list / tuple → 回傳其總和（支援多 dataloader 情況）
        2. 如果是 float('inf') → 回傳 None 代表「未知長度」
        3. 其它型別（int / None）原樣回傳
        """
        if total is None:
            return None
        if isinstance(total, (list, tuple)):
            # 過濾掉 None / inf，避免 sum() 出錯
            try:
                total = sum(int(t)
                            for t in total if t not in (None, float("inf")))
            except (TypeError, ValueError):
                total = None
        elif total == float("inf"):
            total = None
        return int(total) if total is not None else None

    def create_tqdm(
        self,
        desc: str,
        leave: bool,
        position_offset: int = 0,
        total: Any = None,
    ) -> Tqdm:
        """
        Instantiate a Tqdm bar with unified styling.
        """
        position = 2 * self.process_position + position_offset
        total = self._resolve_total(total)

        return Tqdm(
            desc=desc,
            position=position,
            disable=self.is_disabled,
            leave=leave,
            dynamic_ncols=True,
            file=sys.stdout,
            unit_scale=self.unit_scale,
            total=total,
        )

    # --------------------------------------------------------------------- #
    # Lightning‑hooked factory methods
    # --------------------------------------------------------------------- #
    def init_sanity_tqdm(self) -> Tqdm:
        return self.create_tqdm(
            desc=self.sanity_check_description,
            leave=False,
            total=getattr(self.trainer, "num_sanity_val_steps", None),
        )

    def init_train_tqdm(self) -> Tqdm:
        return self.create_tqdm(
            desc=self.train_description,
            leave=True,
            total=getattr(self.trainer, "num_training_batches", None),
        )

    def init_validation_tqdm(self) -> Tqdm:
        has_main_bar = self.trainer.state.fn != "validate"
        return self.create_tqdm(
            desc=self.validation_description,
            leave=not has_main_bar,
            position_offset=int(has_main_bar),
            total=getattr(self.trainer, "num_val_batches", None),
        )

    def init_test_tqdm(self) -> Tqdm:
        return self.create_tqdm(
            desc=self.test_description,
            leave=True,
            total=getattr(self.trainer, "num_test_batches", None),
        )

    def init_predict_tqdm(self) -> Tqdm:
        return self.create_tqdm(
            desc=self.predict_description,
            leave=True,
            total=getattr(self.trainer, "num_predict_batches", None),
        )
