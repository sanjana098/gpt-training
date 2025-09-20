import io
import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import boto3
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class S3MMapConfig:
    bucket: str
    region: str
    manifest_key: str
    local_cache_dir: str
    seq_len: int
    batch_size_per_device: int
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


class MMapShardDataset(Dataset):
    def __init__(self, local_files: List[str], seq_len: int) -> None:
        self.local_files = local_files
        self.seq_len = seq_len
        self.index: List[Tuple[int, int]] = []  # (file_idx, sample_idx)
        for fi, path in enumerate(self.local_files):
            size_bytes = os.path.getsize(path)
            num_int32 = size_bytes // 4
            assert num_int32 % seq_len == 0
            num_samples = num_int32 // seq_len
            for s in range(num_samples):
                self.index.append((fi, s))
        self._mmaps: List[np.memmap] = []

    def _ensure_open(self):
        if self._mmaps:
            return
        self._mmaps = []
        for path in self.local_files:
            mmap = np.memmap(path, mode="r", dtype=np.int32)
            self._mmaps.append(mmap)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._ensure_open()
        fi, si = self.index[idx]
        offset = si * self.seq_len
        arr = self._mmaps[fi][offset : offset + self.seq_len]
        t = torch.from_numpy(np.array(arr, dtype=np.int64))
        return {"input_ids": t}


class S3MMapDataModule(pl.LightningDataModule):
    def __init__(self, cfg: S3MMapConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.s3 = boto3.client("s3", region_name=cfg.region)
        os.makedirs(cfg.local_cache_dir, exist_ok=True)

    def _download_if_needed(self, key: str) -> str:
        local_path = os.path.join(self.cfg.local_cache_dir, key.replace("/", "_"))
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(self.cfg.bucket, key, local_path)
        return local_path

    def setup(self, stage=None):
        # Download manifest
        obj = self.s3.get_object(Bucket=self.cfg.bucket, Key=self.cfg.manifest_key)
        manifest = json.loads(obj["Body"].read())
        train_keys = manifest.get("train", [])
        val_keys = manifest.get("val", [])
        train_local = [self._download_if_needed(k) for k in train_keys]
        val_local = [self._download_if_needed(k) for k in val_keys]
        self.train_dataset = MMapShardDataset(train_local, self.cfg.seq_len)
        self.val_dataset = MMapShardDataset(val_local, self.cfg.seq_len) if val_local else self.train_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size_per_device,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        val_cfg = self.cfg
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size_per_device,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
        )
