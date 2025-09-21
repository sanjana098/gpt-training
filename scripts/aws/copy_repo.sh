#!/usr/bin/env bash
set -euo pipefail

# Copies the current repo to /opt/project on each host and runs bootstrap.
# Usage: scripts/aws/copy_repo.sh --key gpt2-key.pem --hosts cluster_hosts.txt

KEY="gpt2-key.pem"
HOSTS_FILE="cluster_hosts.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --key) KEY="$2"; shift 2 ;;
    --hosts) HOSTS_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ ! -f "$HOSTS_FILE" ]]; then
  echo "hosts file not found: $HOSTS_FILE" >&2
  exit 1
fi

while read -r PUB PRIV IID; do
  echo "Copying to $PUB ($IID)"
  rsync -az -e "ssh -o StrictHostKeyChecking=no -i $KEY" ./ ubuntu@${PUB}:/opt/project
  ssh -o StrictHostKeyChecking=no -i "$KEY" ubuntu@${PUB} 'cd /opt/project && bash scripts/ec2_bootstrap.sh'
done < "$HOSTS_FILE"

echo "Done. Now run launch_torchrun.sh on both nodes with the master's private IP."
