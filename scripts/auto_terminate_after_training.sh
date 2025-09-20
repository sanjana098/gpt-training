#!/usr/bin/env bash
set -euo pipefail

# This script is intended to be run after training completes on each node.
# It syncs TensorBoard logs and checkpoints to S3, then optionally terminates the instance.

AUTO_TERMINATE=${AUTO_TERMINATE:-"false"}
REGION=${AWS_DEFAULT_REGION:-"us-east-2"}
BUCKET=${S3_BUCKET:-"gpt_data"}
RUN_NAME=${RUN_NAME:-"gpt2_124m_pile"}

# Sync logs and checkpoints
aws s3 sync tb_logs s3://${BUCKET}/logs/${RUN_NAME}/ --region ${REGION} || true
aws s3 sync .ckpts s3://${BUCKET}/checkpoints/${RUN_NAME}/ --region ${REGION} || true

if [[ "${AUTO_TERMINATE}" == "true" ]]; then
  INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
  echo "Auto-terminating instance ${INSTANCE_ID} in ${REGION}..."
  aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" || true
fi
