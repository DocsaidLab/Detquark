from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import lightning.pytorch.loggers as pl_loggers
import natsort
import torch.nn as nn
from capybara import (PowerDict, colorstr, dump_json, get_curdir, get_files,
                      get_gpu_cuda_versions, get_package_versions,
                      get_system_info, now)
from torch.utils.data import DataLoader

from .custom_callbacks import CustomTQDMProgressBar

__all__ = [
    'load_model_from_config',
    'build_logger',
    'build_callbacks',
    'build_dataloaders',
    'build_trainer'
]

pl_callbacks.CustomTQDMProgressBar = CustomTQDMProgressBar


def load_model_from_config(
    cfg_name: Union[str, Path],
    root: Union[str, Path] = None,
    stem: Union[str, Path] = None,
    network: Dict[str, Any] = {},
) -> Tuple[nn.Module, PowerDict]:
    """
    Load and return a model instance along with its configuration.

    Args:
        cfg_name: Name of the YAML config file (without .yaml extension).
        root: Base directory where configs are stored. Defaults to this script's directory.
        stem: Optional subdirectory under root where the config resides.
        network: Mapping of model class names to their constructor or LightningModule classes.

    Returns:
        model: Instantiated or restored model.
        cfg: Loaded PowerDict configuration.
    """
    # 1. Determine paths and timestamp
    timestamp = now(fmt="%Y-%m-%d-%H-%M-%S")
    base_dir = Path(root) if root else get_curdir(__file__)
    config_dir = base_dir / (stem or "")
    config_path = config_dir / f"{cfg_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 2. Load YAML config
    cfg: PowerDict = PowerDict.load_yaml(config_path)
    restore_index = cfg.common.get("restore_ind") or timestamp
    cfg.update({'name': str(cfg_name), 'name_ind': restore_index})

    # 3. Verify required fields
    if "common" not in cfg or "model" not in cfg:
        raise KeyError(
            "Config must contain 'common', 'model' keys")

    # 4. Set up experiment identifiers
    cfg.common["name"] = str(cfg_name)
    cfg.common["name_ind"] = restore_index

    # 5. Locate model class
    model_name = cfg.model.name
    ModelClass = getattr(network, model_name)
    if ModelClass is None:
        raise KeyError(f"Model '{model_name}' not found in network mapping")

    # 6. Restore from checkpoint if requested
    if cfg.common.get("is_restore", False):
        ckpt_dir = (base_dir / cfg_name / restore_index /
                    "checkpoint" / "model").resolve()
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {ckpt_dir}")

        # find latest checkpoint if none specified
        ckpt_file = cfg.common.get("restore_ckpt")
        if not ckpt_file:
            candidates = [p for p in get_files(
                ckpt_dir, suffix=[".ckpt"]) if "last" in p.stem]
            if not candidates:
                raise FileNotFoundError(
                    f"No checkpoint files containing 'last' found in {ckpt_dir}")
            ckpt_file = str(natsort.os_sorted(candidates)[-1])

        ckpt_path = ckpt_dir / ckpt_file
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Resolved checkpoint not found: {ckpt_path}")

        print(
            f"[INFO] Loading model from checkpoint: {colorstr(str(ckpt_path))}")
        model = ModelClass.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            cfg=cfg,
            strict=False
        )

        if cfg.common.get("restore_all_states", False):
            cfg.common["checkpoint_path"] = str(ckpt_path)

    # 7. Otherwise instantiate a new model
    else:
        model = ModelClass(cfg=cfg)
        print(
            f"[INFO] Initialized new model '{model_name}' with config '{cfg_name}.yaml'")

    return model, cfg


def build_logger(cfg):
    """
    Create and return a Lightning logger based on `cfg.logger`, and
    save the full configuration to a JSON file in the logger's save directory.

    Args:
        cfg (PowerDict): Configuration object.
            - cfg.logger.name (str): Logger class name in lightning.pytorch.loggers
              (e.g., 'TensorBoardLogger', 'CSVLogger').
            - cfg.logger.options (dict): Keyword arguments for the logger constructor.
              Must include 'save_dir'.

    Returns:
        logger: Instantiated Lightning logger.
    """
    # Extract logger name and options
    logger_name = cfg.logger.get("name")
    options = cfg.logger.get("options", {})

    # Validate logger class
    if not logger_name or not hasattr(pl_loggers, logger_name):
        raise KeyError(
            f"Logger '{logger_name}' not found in lightning.pytorch.loggers")

    # Validate save directory
    save_dir = options.get("save_dir")
    if not save_dir:
        raise KeyError("cfg.logger.options must include 'save_dir'")

    # Ensure the save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Instantiate the logger
    LoggerCls = getattr(pl_loggers, logger_name)
    logger = LoggerCls(**options)
    print(f"[INFO] Initialized logger '{logger_name}'.")
    print(f"[INFO] Saving configuration to {save_path / 'config.json'}")

    # Dump the full cfg to JSON for reproducibility
    dump_json(cfg, save_path / "config.json")
    print("[INFO] Configuration file saved.")

    return logger


def build_callbacks(cfg: PowerDict) -> List[pl_callbacks.Callback]:
    """
    Instantiate and return a list of Lightning callbacks based on the config.

    Args:
        cfg (PowerDict): Configuration dict containing:
            - root_dir (str): Base directory for outputs.
            - common.batch_size (int): Global batch size, used for progress bar scaling.
            - callbacks (List[dict]): Each dict must have:
                - name (str): Class name of the callback in lightning.pytorch.callbacks
                - options (dict): Keyword args for that callback.

    Returns:
        List[pl_callbacks.Callback]: Initialized callback instances.
    """
    callbacks: List[pl_callbacks.Callback] = []
    root_dir = Path(cfg.get("root_dir", "."))

    for cb_cfg in cfg.get("callbacks", []):
        name = cb_cfg.get("name")
        options = cb_cfg.get("options", {}).copy()

        # Validate callback class exists
        if not hasattr(pl_callbacks, name):
            raise KeyError(
                f"Callback '{name}' not found in lightning.pytorch.callbacks")

        # Special handling for ModelCheckpoint
        if name == "ModelCheckpoint":
            ckpt_dir = root_dir / "checkpoint" / "model"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # only set dirpath if not already specified
            options.update({"dirpath": str(ckpt_dir)})

        # Special handling for TQDM progress bar scaling
        if name == "CustomTQDMProgressBar":
            batch_size = cfg.common.get("batch_size", 1)
            options.update({"unit_scale": batch_size})

        # Instantiate the callback
        CallbackClass = getattr(pl_callbacks, name)
        callback = CallbackClass(**options)
        print(f"[INFO] Registered callback: {name} with options {options}")

        callbacks.append(callback)

    return callbacks


def build_dataloaders(
    cfg: PowerDict,
    dataset_mapping: Dict[str, Any],
    collate_fn: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct training and validation DataLoaders from config and dataset mapping.

    Args:
        cfg (PowerDict): Configuration containing:
            - common.batch_size (int)
            - dataset.train_options:
                name: str                 # key in dataset_mapping
                options: dict             # kwargs for the training Dataset constructor
            - dataset.valid_options: same structure as train_options
            - dataloader.train_options: dict of DataLoader kwargs
            - dataloader.valid_options: dict of DataLoader kwargs
            - global_settings (optional): dict to merge into both dataset options
        dataset_mapping (Dict[str, Any]): Maps dataset names to their classes/constructors.
        collate_fn (Callable, optional): Custom collate function to pass to DataLoader.

    Returns:
        train_loader (DataLoader), valid_loader (DataLoader)
    """
    # 1. Prepare DataLoader configs, ensure batch_size is set
    train_loader_cfg = dict(cfg.dataloader.train_options)
    valid_loader_cfg = dict(cfg.dataloader.valid_options)
    train_loader_cfg.update({'batch_size': cfg.common.batch_size})
    valid_loader_cfg.update({'batch_size': cfg.common.batch_size})

    # 2. Prepare Dataset configs
    train_ds_name = cfg.dataset.train_options.name
    train_ds_opts = dict(cfg.dataset.train_options.options)
    valid_ds_name = cfg.dataset.valid_options.name
    valid_ds_opts = dict(cfg.dataset.valid_options.options)

    # 3. Merge global settings if provided
    if "global_settings" in cfg:
        train_ds_opts.update(cfg.global_settings)
        valid_ds_opts.update(cfg.global_settings)

    # 4. Instantiate Datasets
    TrainDataset = getattr(dataset_mapping, train_ds_name)
    if TrainDataset is None:
        raise KeyError(
            f"Training dataset '{train_ds_name}' not found in mapping")
    ValidDataset = getattr(dataset_mapping, valid_ds_name)
    if ValidDataset is None:
        raise KeyError(
            f"Validation dataset '{valid_ds_name}' not found in mapping")

    train_dataset = TrainDataset(**train_ds_opts)
    valid_dataset = ValidDataset(**valid_ds_opts)

    # 5. Build DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        **train_loader_cfg
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=collate_fn,
        **valid_loader_cfg
    )

    return train_loader, valid_loader


def build_trainer(
    cfg: PowerDict,
    root: Union[str, Path] = None
) -> pl.Trainer:
    """
    Initialize and return a Lightning Trainer configured by `cfg`.

    This function sets up:
      1. Experiment directory structure under `root` or script folder.
      2. Logger save directory and instantiates the logger.
      3. Callbacks as defined in cfg.
      4. Dumps environment metadata (packages, GPU/CUDA, system) to JSON files.

    Args:
        cfg (PowerDict): Experiment config containing:
            - name (str): Experiment name.
            - name_ind (str): Run identifier.
            - common and other sections.
            - logger.options.save_dir (str): Relative path under experiment dir.
            - trainer (dict): kwargs for pl.Trainer.
        root (str|Path, optional): Base path for experiments. Defaults to this script's dir.

    Returns:
        pl.Trainer: Configured Lightning Trainer.
    """
    # 1. Determine base and experiment directories
    base_dir = Path(root) if root else get_curdir(__file__)
    experiment_dir = base_dir / cfg.name / cfg.name_ind

    # 2. Configure and create logging directory
    log_dir = experiment_dir / cfg.logger.options.get("save_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.common["root_dir"] = str(experiment_dir)
    cfg.logger.options["save_dir"] = str(log_dir)
    print(f"[INFO] Experiment directory: {experiment_dir}")
    print(f"[INFO] Log directory: {log_dir}")

    # 3. Build callbacks and logger
    callbacks = build_callbacks(cfg)
    logger = build_logger(cfg)
    print(
        f"[INFO] Registered {len(callbacks)} callbacks and logger '{cfg.logger.name}'.")

    # 4. Dump environment metadata for reproducibility
    metadata = {
        "package_versions.json": get_package_versions(),
        "gpu_cuda_versions.json": get_gpu_cuda_versions(),
        "system_info.json": get_system_info(),
    }
    for filename, info in metadata.items():
        filepath = log_dir / filename
        dump_json(info, filepath)
        print(f"[INFO] Saved metadata file: {filepath}")

    # 5. Instantiate and return the Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer
    )
    print("[INFO] Lightning Trainer initialized successfully.")
    return trainer
