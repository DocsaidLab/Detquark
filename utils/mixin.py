from pathlib import Path
from typing import Any, Dict, List

import lightning as L
from chameleon import build_optimizer


class BaseMixin(L.LightningModule):

    def apply_solver_config(
        self,
        optimizer: Dict[str, Any],
        lr_scheduler: Dict[str, Any]
    ) -> None:
        self.optimizer_name, self.optimizer_opts = optimizer.values()
        self.sche_name, self.sche_opts, self.sche_pl_opts = lr_scheduler.values()

    def get_optimizer_params(self) -> List[Dict[str, Any]]:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)],
                "weight_decay": self.optimizer_opts.get('weight_decay', 0.0),
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_optimizers(self):
        optimizer = build_optimizer(
            name=self.optimizer_name,
            params=self.get_optimizer_params(),
            **self.optimizer_opts
        )

        scheduler = build_optimizer(
            name=self.sche_name,
            optimizer=optimizer,
            **self.sche_opts
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                **self.sche_pl_opts
            }
        }

    def get_lr(self):
        return self.trainer.optimizers[0].param_groups[0]['lr']

    @property
    def preview_dir(self) -> Path:
        img_path = Path(self.cfg.root_dir) / "preview" / \
            f'epoch_{self.current_epoch}'
        if not img_path.exists():
            img_path.mkdir(parents=True)
        return img_path
