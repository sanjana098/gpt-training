import os
from dataclasses import dataclass
from typing import Optional, Tuple

import boto3


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://"), f"Invalid S3 URI: {uri}"
    path = uri[5:]
    parts = path.split("/", 1)
    bucket = parts[0]
    key_prefix = parts[1] if len(parts) > 1 else ""
    return bucket, key_prefix


@dataclass
class S3SyncConfig:
    region: str
    logs_s3_prefix: str
    ckpt_s3_prefix: str
    local_logs_dir: str = "tb_logs"
    local_ckpt_dir: str = ".ckpts"


class S3Sync:
    def __init__(self, cfg: S3SyncConfig) -> None:
        self.cfg = cfg
        self.client = boto3.client("s3", region_name=cfg.region)
        self.logs_bucket, self.logs_prefix = parse_s3_uri(cfg.logs_s3_prefix)
        self.ckpt_bucket, self.ckpt_prefix = parse_s3_uri(cfg.ckpt_s3_prefix)

    def _upload_dir(self, local_dir: str, bucket: str, prefix: str) -> None:
        if not os.path.isdir(local_dir):
            return
        for root, _, files in os.walk(local_dir):
            for f in files:
                local_path = os.path.join(root, f)
                rel = os.path.relpath(local_path, local_dir)
                key = f"{prefix.rstrip('/')}/{rel}"
                self.client.upload_file(local_path, bucket, key)

    def upload_logs(self) -> None:
        self._upload_dir(self.cfg.local_logs_dir, self.logs_bucket, self.logs_prefix)

    def upload_ckpts(self) -> None:
        self._upload_dir(self.cfg.local_ckpt_dir, self.ckpt_bucket, self.ckpt_prefix)


# Lightning callback
import pytorch_lightning as pl


class S3SyncCallback(pl.Callback):
    def __init__(self, sync: S3Sync, every_val: bool = True) -> None:
        super().__init__()
        self.sync = sync
        self.every_val = every_val

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.every_val:
            return
        if trainer.global_rank == 0:
            self.sync.upload_logs()
            self.sync.upload_ckpts()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            self.sync.upload_logs()
            self.sync.upload_ckpts()
