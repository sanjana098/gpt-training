import argparse
import os

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
