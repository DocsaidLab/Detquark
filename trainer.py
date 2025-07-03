import json
import textwrap
from pathlib import Path

import cv2
import torch
from capybara import dump_json, get_curdir
from chameleon import calculate_flops
from rich.console import Console
from rich.table import Table

from . import dataset as ds
from . import model as net
from .dataset import coco_collate_fn
from .utils import build_dataloaders, build_trainer, load_model_from_config

DIR = get_curdir(__file__)

# Set threading and precision for deterministic performance
cv2.setNumThreads(0)
torch.set_num_threads(1)
torch.set_float32_matmul_precision('medium')


def print_meta_data_rich(meta_data):
    console = Console()
    console.rule("[bold cyan]MODEL META DATA")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")
    for k, v in meta_data.items():
        table.add_row(k, str(v))
    console.print(table)
    console.rule()


def main_train(cfg_name: str) -> None:
    """
    Main entry point for training.

    This function performs the following steps:
      1. Loads model and configuration.
      2. Builds training and validation data loaders.
      3. Adjusts scheduler warmup for PolynomialLRWarmup.
      4. Initializes the Lightning Trainer.
      5. Updates .gitignore to exclude experiment directory.
      6. Computes and records model FLOPs, MACs, and parameter count.
      7. Starts the training loop.

    Args:
        cfg_name (str): Base name of the YAML config file (without .yaml).
    """
    # 1. Load model and config
    model, cfg = load_model_from_config(
        cfg_name=cfg_name,
        root=DIR,
        stem='config',
        network=net
    )

    # 2. Build data loaders
    train_loader, valid_loader = build_dataloaders(
        cfg, ds, collate_fn=coco_collate_fn)

    # 3. Adjust warmup for PolynomialLRWarmup scheduler
    if cfg.lr_scheduler.name == 'PolynomialLRWarmup':
        max_epochs = cfg.trainer.max_epochs
        batch_size = cfg.common.batch_size
        accumulate = cfg.trainer.accumulate_grad_batches
        total_samples = len(train_loader.dataset)
        total_iters = (max_epochs * total_samples) // (batch_size * accumulate)
        warmup_iters = int(0.1 * total_iters)
        cfg.lr_scheduler.options.update(
            warmup_iters=warmup_iters,
            total_iters=total_iters
        )

    # 4. Initialize Trainer
    trainer = build_trainer(cfg, root=DIR)

    # 5. Ensure experiment directory is ignored by Git
    gitignore_path = DIR / '.gitignore'
    ignore_entry = f"{cfg.name}/"
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        if ignore_entry not in content:
            gitignore_path.write_text(content + f"\n{ignore_entry}\n")
            print(f"[INFO] Added '{ignore_entry}' to .gitignore")

    # 6. Compute and record model meta data
    log_dir = Path(cfg.logger.options.save_dir)
    flops, macs, params = calculate_flops(
        model,
        input_shape=(1, 3, *cfg.global_settings.image_size),
        print_detailed=False,
        print_results=False
    )
    meta_data = {'FLOPs': flops, 'MACs': macs, 'Params': params}
    dump_json(meta_data, log_dir / 'model_meta_data.json')
    print_meta_data_rich(meta_data)

    # 7. Start training
    checkpoint = cfg.common.checkpoint_path if getattr(
        cfg.common, 'restore_all_states', False) else None
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=checkpoint
    )
