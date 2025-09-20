EC2 + torchrun instructions (Option B hardware: 2× g5.12xlarge, us-east-2)

1. Create 2 spot instances with AMI: Deep Learning AMI (Ubuntu 22.04), instance type g5.12xlarge in us-east-2.
2. Attach an IAM role with S3 read/write access to bucket `gpt_data`.
3. On each node:
```bash
sudo apt update && sudo apt install -y python3-venv
# Sync project to /opt/project (use scp/rsync/git clone)
bash scripts/ec2_bootstrap.sh
```
4. Pick one node as rendezvous host, note its private IP, open port 29400 on the security group.
5. On every node, run:
```bash
export CKPT_DIR=/opt/project/.ckpts
# Node 0 (master):
bash scripts/launch_torchrun.sh 2 4 <MASTER_PRIVATE_IP>:29400 --config configs/config.yaml

# Node 1 (worker):
bash scripts/launch_torchrun.sh 2 4 <MASTER_PRIVATE_IP>:29400 --config configs/config.yaml
```
6. Monitor logs with `tail -f` on tb_logs and CloudWatch (if enabled at instance level).
