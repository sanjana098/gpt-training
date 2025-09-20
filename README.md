# LLM Pretraining Pipeline (EC2 + torchrun)

This repository contains a modular pipeline to pretrain GPT-2 124M on The Pile using PyTorch Lightning, with flexible distributed strategies (DDP/FSDP/DeepSpeed ZeRO-2/3 and 2D TP+FSDP), S3 sharded data, and EC2 + torchrun orchestration.

## Quickstart

1. Install dependencies:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data shards to S3 (The Pile → tokenized, packed, sharded):
```bash
python src/scripts/prepare_data.py --bucket gpt_data --region us-east-2 \
  --seq_len 1024 --num_tokens 2500000000 --shard_tokens 20000000 --tokenizer gpt2
```

3. Configure training strategy in `configs/train/*.yaml`. Default is `tp_fsdp` with `tp_size=2`.

4. Launch multi-node training (example 2 nodes × 4 GPUs):
```bash
bash scripts/launch_torchrun.sh 2 4 host0:29400
```

## Configs
- `configs/model/gpt2_124m.yaml`: GPT-2 model shape
- `configs/data/pile_seq1024.yaml`: dataset/storage/batch params
- `configs/train/*.yaml`: distributed strategy and trainer parameters
- `configs/aws/ec2.yaml`: EC2 cluster parameters

## Notes
- Observability: TensorBoard (local) and CloudWatch via EC2 logs.
- Checkpoints saved to `.ckpts/` unless `CKPT_DIR` is set (can point to an S3 mounted path).
- Tokenizer: GPT-2 BPE (`gpt2`). Optional tokenizer training can be added later.
