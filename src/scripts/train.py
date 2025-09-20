import os
import math
import argparse

import pytorch_lightning as pl
import sys
from pathlib import Path

# Ensure 'src' is on sys.path when running as a script
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from omegaconf import OmegaConf

from training.strategy_factory import make_strategy
from training.s3_sync import S3Sync, S3SyncConfig, S3SyncCallback
from llm_core.models.gpt2_lightning import GPT2LightningConfig, GPT2LightningModule
from llm_core.models.gpt2_custom import GPT2CustomConfig, GPT2CustomLightningModule
from llm_core.modules.base_module import OptimizerConfig
from data_core.s3_mmap_datamodule import S3MMapConfig, S3MMapDataModule


def build_trainer(cfg):
    precision = cfg.train.precision
    strategy = make_strategy(cfg.train.strategy, precision, cfg.train)
    max_steps = cfg.train.max_steps
    logger = TensorBoardLogger("tb_logs", name=cfg.run_name)
    ckpt = ModelCheckpoint(
        dirpath=os.environ.get("CKPT_DIR", ".ckpts"),
        save_last=True,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        every_n_train_steps=10_000,
        filename="{step}-{val_loss:.3f}",
    )
    lrmon = LearningRateMonitor(logging_interval="step")
    # S3 sync callback
    bucket = cfg.data.bucket
    region = cfg.data.region
    run_name = cfg.run_name
    logs_prefix = f"s3://{bucket}/logs/{run_name}/"
    ckpt_prefix = f"s3://{bucket}/checkpoints/{run_name}/"
    s3sync = S3Sync(S3SyncConfig(region=region, logs_s3_prefix=logs_prefix, ckpt_s3_prefix=ckpt_prefix))
    s3cb = S3SyncCallback(s3sync, every_val=True)
    trainer = pl.Trainer(
        strategy=strategy,
        precision=precision,
        max_steps=max_steps,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        logger=logger,
        callbacks=[ckpt, lrmon, s3cb],
    )
    return trainer


def _compose_config(config_path: str):
    base = OmegaConf.load(config_path)
    # If using Hydra-like defaults, compose manually
    if "defaults" in base:
        root = Path(config_path).resolve().parent
        def _load_rel(*parts):
            return OmegaConf.load(str(root / Path(*parts)))
        composed = OmegaConf.create({})
        # model
        for item in base["defaults"]:
            if isinstance(item, dict):
                if "model" in item:
                    composed.model = _load_rel("model", f"{item['model']}.yaml")
                if "data" in item:
                    composed.data = _load_rel("data", f"{item['data']}.yaml")
                if "train" in item:
                    composed.train = _load_rel("train", f"{item['train']}.yaml")
                if "aws" in item:
                    composed.aws = _load_rel("aws", f"{item['aws']}.yaml")
                if "exp" in item:
                    # optional layer for overrides
                    pass
        # Merge top-level keys like seed/run_name
        for k in base.keys():
            if k != "defaults":
                composed[k] = base[k]
        return composed
    else:
        return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = _compose_config(args.config)

    # Data manifest key depends on tokenizer path used during prep
    tok_id = "gpt2"  # initial default
    manifest_key = f"data/the_pile/{tok_id}/seq{cfg.data.seq_len}/manifest.json"

    data_cfg = S3MMapConfig(
        bucket=cfg.data.bucket,
        region=cfg.data.region,
        manifest_key=manifest_key,
        local_cache_dir=cfg.data.local_cache_dir,
        seq_len=cfg.data.seq_len,
        batch_size_per_device=cfg.data.batch_size_per_device,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=cfg.data.persistent_workers,
    )
    dm = S3MMapDataModule(data_cfg)

    use_tp = cfg.train.strategy == "tp_fsdp" and cfg.train.parallel.tp_size and int(cfg.train.parallel.tp_size) > 1
    if use_tp:
        model_cfg = GPT2CustomConfig(
            vocab_size=cfg.model.vocab_size,
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
            n_embd=cfg.model.n_embd,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
            tp_size=int(cfg.train.parallel.tp_size),
        )
    else:
        model_cfg = GPT2LightningConfig(
            vocab_size=cfg.model.vocab_size,
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
            n_embd=cfg.model.n_embd,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
        )
    # Convert target tokens to steps using batch size and world size
    global_batch_tokens = cfg.data.batch_size_per_device * cfg.data.seq_len
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_tokens *= world_size
    steps_target = math.ceil(cfg.data.num_tokens_target / global_batch_tokens)

    optim_cfg = OptimizerConfig(max_steps=steps_target)
    model = GPT2CustomLightningModule(model_cfg, optim_cfg) if use_tp else GPT2LightningModule(model_cfg, optim_cfg)

    trainer = build_trainer(cfg)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
