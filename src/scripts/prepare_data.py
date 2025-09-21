import argparse
import os
import sys
from pathlib import Path

# Ensure 'src' is on sys.path when running as a script
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from omegaconf import OmegaConf
from data_core.pile_prep import PilePrepConfig, prepare_and_upload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", default="configs/data/pile_seq1024.yaml")
    # Optional overrides
    p.add_argument("--bucket", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--num_tokens", type=int, default=None)
    p.add_argument("--tokenizer", default="gpt2")
    args = p.parse_args()

    data_cfg = OmegaConf.load(args.data_config)

    bucket = args.bucket or data_cfg.get("bucket")
    region = args.region or data_cfg.get("region", "us-east-2")
    seq_len = int(data_cfg.get("seq_len", 1024))
    shard_tokens = int(data_cfg.get("shard_num_tokens", 20_000_000))
    val_fraction = float(data_cfg.get("val_fraction", 0.005))
    local_cache_dir = data_cfg.get("local_cache_dir", "/tmp/pile_prep")
    num_tokens_target = args.num_tokens or int(data_cfg.get("num_tokens_target", 2_500_000_000))

    if not bucket:
        raise SystemExit("Bucket must be set in data config or via --bucket override")

    cfg = PilePrepConfig(
        bucket=bucket,
        region=region,
        seq_len=seq_len,
        shard_num_tokens=shard_tokens,
        val_fraction=val_fraction,
        tokenizer_name=args.tokenizer,
        local_cache_dir=local_cache_dir,
    )
    prepare_and_upload(cfg, num_tokens_target=num_tokens_target)


if __name__ == "__main__":
    main()
