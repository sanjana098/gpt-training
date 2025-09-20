import argparse
import os
import sys
from pathlib import Path

# Ensure 'src' is on sys.path when running as a script
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_core.pile_prep import PilePrepConfig, prepare_and_upload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_tokens", type=int, default=2_500_000_000)
    p.add_argument("--shard_tokens", type=int, default=20_000_000)
    p.add_argument("--val_fraction", type=float, default=0.005)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--local_cache_dir", default="/tmp/pile_prep")
    args = p.parse_args()

    cfg = PilePrepConfig(
        bucket=args.bucket,
        region=args.region,
        seq_len=args.seq_len,
        shard_num_tokens=args.shard_tokens,
        val_fraction=args.val_fraction,
        tokenizer_name=args.tokenizer,
        local_cache_dir=args.local_cache_dir,
    )
    prepare_and_upload(cfg, num_tokens_target=args.num_tokens)


if __name__ == "__main__":
    main()
