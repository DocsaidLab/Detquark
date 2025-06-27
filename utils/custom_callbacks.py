import sys

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
        unit_scale: float = 1.0,
        *args,
        **kwargs
    ):
        """
        Args:
            unit_scale: Factor to scale the progress "unit" (e.g., samples per iteration).
            *args, **kwargs: Passed to base TQDMProgressBar initializer.
        """
        super().__init__(*args, **kwargs)
        self.unit_scale = unit_scale

    def create_tqdm(
        self,
        desc: str,
        leave: bool,
        position_offset: int = 0,
        total: int | None = None,
    ) -> Tqdm:
        """
        Internal helper to instantiate a Tqdm bar with standard settings.

        Args:
            desc: Description label for the progress bar.
            leave: Whether to leave the bar displayed after completion.
            position_offset: Additional offset for bar positioning.
            total: Optional total number of steps for the bar.

        Returns:
            Configured Tqdm instance.
        """
        position = 2 * self.process_position + position_offset
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

    def init_sanity_tqdm(self) -> Tqdm:
        return self.create_tqdm(
            desc=self.sanity_check_description,
            leave=False,
            total=getattr(self.trainer, 'num_sanity_val_steps', None),
        )

    def init_train_tqdm(self) -> Tqdm:
        return self.create_tqdm(
            desc=self.train_description,
            leave=True,
            total=getattr(self.trainer, 'num_training_batches', None),
        )

    def init_validation_tqdm(self) -> Tqdm:
        has_main_bar = self.trainer.state.fn != "validate"
        return self.create_tqdm(
            desc=self.validation_description,
            leave=not has_main_bar,
            position_offset=int(has_main_bar),
            total=getattr(self.trainer, 'num_val_batches', None),
        )

    def init_test_tqdm(self) -> Tqdm:
        return self.create_tqdm(
            desc=self.test_description,
            leave=True,
            total=getattr(self.trainer, 'num_test_batches', None),
        )

    def init_predict_tqdm(self) -> Tqdm:
        return self.create_tqdm(
            desc=self.predict_description,
            leave=True,
            total=getattr(self.trainer, 'num_predict_batches', None),
        )
