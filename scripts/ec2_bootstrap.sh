#!/usr/bin/env bash
set -euo pipefail

# Basic bootstrap: install CUDA toolkit deps, Python, and dependencies. Assumes DLAMI is used.

sudo mkdir -p /opt/project
sudo chown $USER:$USER /opt/project

# Optional: create venv
python3 -m venv /opt/project/venv
source /opt/project/venv/bin/activate
pip install --upgrade pip
pip install -r /opt/project/requirements.txt
pip install awscli

# EFA not enabled here; for p4/p5 you can extend with EFA setup.

mkdir -p /opt/project/tb_logs /opt/project/.ckpts
