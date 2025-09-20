import io
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import boto3
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from tqdm import tqdm


@dataclass
class PilePrepConfig:
    bucket: str
    region: str
    seq_len: int
    shard_num_tokens: int
    val_fraction: float
    tokenizer_name: str = "gpt2"
    dataset_name: str = "monology/pile-uncopyrighted"
    local_cache_dir: str = "/tmp/pile_prep"
    prefix: str = "data/the_pile"


def iter_documents(dataset_name: str) -> Iterable[str]:
    ds = load_dataset(dataset_name, split="train", streaming=True)
    for ex in ds:
        text = ex.get("text")
        if text and isinstance(text, str) and len(text) > 0:
            yield text


def build_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    tok.model_max_length = 1_000_000_000
    tok.padding_side = "right"
    return tok


def pack_tokens(token_ids: List[int], seq_len: int) -> np.ndarray:
    n = len(token_ids)
    n_trim = n - (n % seq_len)
    if n_trim == 0:
        return np.empty((0, seq_len), dtype=np.int32)
    arr = np.array(token_ids[:n_trim], dtype=np.int32)
    return arr.reshape(-1, seq_len)


def upload_to_s3(s3, bucket: str, key: str, data: bytes) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data)


def prepare_and_upload(cfg: PilePrepConfig, num_tokens_target: int) -> None:
    os.makedirs(cfg.local_cache_dir, exist_ok=True)
    s3 = boto3.client("s3", region_name=cfg.region)
    tok = build_tokenizer(cfg.tokenizer_name)

    shard_tokens_target = cfg.shard_num_tokens
    seq_len = cfg.seq_len
    tokens_buffer: List[int] = []
    shard_index = 0
    total_tokens = 0

    train_manifest = []

    for text in tqdm(iter_documents(cfg.dataset_name), desc="tokenizing"):
        ids = tok.encode(text).ids
        tokens_buffer.extend(ids)
        while len(tokens_buffer) >= shard_tokens_target:
            shard_ids = tokens_buffer[:shard_tokens_target]
            tokens_buffer = tokens_buffer[shard_tokens_target:]
            samples = pack_tokens(shard_ids, seq_len)
            if samples.shape[0] == 0:
                continue
            bin_bytes = samples.tobytes()
            key = f"{cfg.prefix}/{tok.name_or_path}/seq{seq_len}/train/shard-{shard_index:06d}.bin"
            upload_to_s3(s3, cfg.bucket, key, bin_bytes)
            shard_index += 1
            total_tokens += samples.size
            train_manifest.append(key)
            if total_tokens >= num_tokens_target:
                break
        if total_tokens >= num_tokens_target:
            break

    # Validation split: take a small extra buffer
    if len(tokens_buffer) >= seq_len:
        samples = pack_tokens(tokens_buffer, seq_len)
        val_tokens = int(samples.size * cfg.val_fraction)
        if val_tokens > 0:
            val_samples = samples[: val_tokens // seq_len]
            if val_samples.shape[0] > 0:
                key = f"{cfg.prefix}/{tok.name_or_path}/seq{seq_len}/val/val-000000.bin"
                upload_to_s3(s3, cfg.bucket, key, val_samples.tobytes())

    # Upload manifest
    manifest = {
        "train": train_manifest,
        "val": [f"{cfg.prefix}/{tok.name_or_path}/seq{seq_len}/val/val-000000.bin"],
        "seq_len": seq_len,
        "tokenizer": tok.name_or_path,
        "num_tokens": total_tokens,
    }
    manifest_key = f"{cfg.prefix}/{tok.name_or_path}/seq{seq_len}/manifest.json"
    upload_to_s3(s3, cfg.bucket, manifest_key, json.dumps(manifest).encode("utf-8"))
    print(f"Uploaded manifest to s3://{cfg.bucket}/{manifest_key}")
