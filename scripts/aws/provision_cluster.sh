#!/usr/bin/env bash
set -euo pipefail

# Provisions 2x g5.12xlarge in us-east-2 with SG, IAM role, key pair, and outputs host info.
# Requires AWS CLI with permissions: ec2, iam, s3 (for head-bucket), sts.
# Usage:
#   scripts/aws/provision_cluster.sh \
#     --region us-east-2 \
#     --bucket <your-s3-bucket> \
#     --key-name gpt2-key \
#     --role-name gpt2-ec2-role \
#     --profile-name gpt2-ec2-profile \
#     --sg-name gpt2-sg \
#     --tag gpt2-train

REGION="us-east-2"
BUCKET=""
KEY_NAME="gpt2-key"
ROLE_NAME="gpt2-ec2-role"
PROFILE_NAME="gpt2-ec2-profile"
SG_NAME="gpt2-sg"
TAG_PREFIX="gpt2-train"
COUNT=2
INST_TYPE="g5.12xlarge"
AMI_ID_OVERRIDE=""
MARKET="spot"  # spot | on-demand

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --key-name) KEY_NAME="$2"; shift 2 ;;
    --role-name) ROLE_NAME="$2"; shift 2 ;;
    --profile-name) PROFILE_NAME="$2"; shift 2 ;;
    --sg-name) SG_NAME="$2"; shift 2 ;;
    --tag) TAG_PREFIX="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --market) MARKET="$2"; shift 2 ;;
    --ami-id) AMI_ID_OVERRIDE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  echo "--bucket is required" >&2
  exit 1
fi

AWS() { aws --region "$REGION" "$@"; }

# Validate bucket exists
if ! AWS s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
  echo "Bucket $BUCKET not found in $REGION or not accessible" >&2
  exit 1
fi

# Get default VPC and its CIDR
VPC_ID=$(AWS ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
if [[ "$VPC_ID" == "None" || -z "$VPC_ID" ]]; then
  echo "No default VPC found in $REGION. Please provide resources manually." >&2
  exit 1
fi
VPC_CIDR=$(AWS ec2 describe-vpcs --vpc-ids "$VPC_ID" --query 'Vpcs[0].CidrBlock' --output text)

# Create or reuse key pair
if ! AWS ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo "Creating key pair $KEY_NAME"
  AWS ec2 create-key-pair --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "$KEY_NAME.pem"
  chmod 400 "$KEY_NAME.pem"
else
  echo "Key pair $KEY_NAME already exists; skipping. Ensure you have the .pem locally."
fi

# Create or reuse security group
SG_ID=$(AWS ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC_ID" --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  echo "Creating security group $SG_NAME in $VPC_ID"
  SG_ID=$(AWS ec2 create-security-group --group-name "$SG_NAME" --description "GPT2 training SG" --vpc-id "$VPC_ID" --query 'GroupId' --output text)
  # Egress all
  # Many accounts have default "allow all egress"; ignore duplicate errors
  AWS ec2 authorize-security-group-egress --group-id "$SG_ID" --ip-permissions IpProtocol=-1,IpRanges='[{CidrIp=0.0.0.0/0}]' || true
  # SSH from caller IP
  MY_IP=$(curl -s https://checkip.amazonaws.com || echo "0.0.0.0")
  AWS ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges="[{CidrIp=${MY_IP}/32,Description=ssh}]" || true
  # NCCL/torchrun rendezvous port from within VPC only
  AWS ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions IpProtocol=tcp,FromPort=29400,ToPort=29400,IpRanges="[{CidrIp=${VPC_CIDR},Description=rdzv}]" || true
else
  echo "Using existing security group $SG_ID"
fi

# Create or reuse IAM role + instance profile with S3 access to bucket
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "Creating IAM role $ROLE_NAME"
  TRUST=$(cat <<'JSON'
{ "Version": "2012-10-17", "Statement": [ { "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole" } ] }
JSON
)
  echo "$TRUST" > /tmp/trust.json
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file:///tmp/trust.json >/dev/null
  POLICY=$(cat <<JSON
{ "Version": "2012-10-17", "Statement": [
  {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":["arn:aws:s3:::$BUCKET"]},
  {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:AbortMultipartUpload","s3:ListBucketMultipartUploads"],"Resource":["arn:aws:s3:::$BUCKET/*"]}
] }
JSON
)
  echo "$POLICY" > /tmp/policy.json
  aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name S3AccessPolicy --policy-document file:///tmp/policy.json >/dev/null
else
  echo "Using existing IAM role $ROLE_NAME"
fi

if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
  echo "Creating instance profile $PROFILE_NAME"
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" >/dev/null || true
fi

find_dlami() {
  # Try multiple patterns/owners to locate a recent DLAMI GPU PyTorch Ubuntu image
  local ami
  # Owner: AWS DLAMI account
  ami=$(AWS ec2 describe-images \
    --owners 898082745236 \
    --filters Name=name,Values='Deep Learning AMI GPU PyTorch *Ubuntu*' Name=state,Values=available \
    --query 'Images | sort_by(@, &CreationDate)[-1].ImageId' --output text)
  if [[ "$ami" != "None" && -n "$ami" ]]; then echo "$ami"; return; fi
  # Owner: amazon shortcut
  ami=$(AWS ec2 describe-images \
    --owners amazon \
    --filters Name=name,Values='Deep Learning AMI GPU PyTorch *Ubuntu*' Name=state,Values=available \
    --query 'Images | sort_by(@, &CreationDate)[-1].ImageId' --output text)
  if [[ "$ami" != "None" && -n "$ami" ]]; then echo "$ami"; return; fi
  # Base DLAMI GPU if PyTorch-specific not found
  ami=$(AWS ec2 describe-images \
    --owners 898082745236 \
    --filters Name=name,Values='Deep Learning Base AMI GPU *Ubuntu*' Name=state,Values=available \
    --query 'Images | sort_by(@, &CreationDate)[-1].ImageId' --output text)
  if [[ "$ami" != "None" && -n "$ami" ]]; then echo "$ami"; return; fi
  echo "None"
}

if [[ -n "$AMI_ID_OVERRIDE" ]]; then
  AMI_ID="$AMI_ID_OVERRIDE"
else
  AMI_ID=$(find_dlami)
fi
if [[ "$AMI_ID" == "None" || -z "$AMI_ID" ]]; then
  echo "Failed to find a suitable Deep Learning AMI in $REGION. Provide --ami-id to override." >&2
  exit 1
fi
echo "Using AMI $AMI_ID"

echo "Launching $COUNT x $INST_TYPE with AMI $AMI_ID in $REGION (market=$MARKET)..."

set +e
if [[ "$MARKET" == "spot" ]]; then
  RUN_OUT=$(AWS ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INST_TYPE" \
    --count $COUNT \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile Name="$PROFILE_NAME" \
    --instance-market-options MarketType=spot \
    --instance-initiated-shutdown-behavior terminate \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_PREFIX}]" \
    --query 'Instances[].InstanceId' --output text 2> /tmp/run_err.txt)
  RC=$?
  if [[ $RC -ne 0 ]]; then
    ERRMSG=$(cat /tmp/run_err.txt)
    echo "Spot launch failed: $ERRMSG" >&2
    echo "Falling back to on-demand for this run..." >&2
    RUN_OUT=$(AWS ec2 run-instances \
      --image-id "$AMI_ID" \
      --instance-type "$INST_TYPE" \
      --count $COUNT \
      --key-name "$KEY_NAME" \
      --security-group-ids "$SG_ID" \
      --iam-instance-profile Name="$PROFILE_NAME" \
      --instance-initiated-shutdown-behavior terminate \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_PREFIX}]" \
      --query 'Instances[].InstanceId' --output text)
    RC=$?
    if [[ $RC -ne 0 ]]; then
      echo "On-demand launch also failed. Consider using --count 1 or a different instance type/region." >&2
      exit 1
    fi
  fi
else
  RUN_OUT=$(AWS ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INST_TYPE" \
    --count $COUNT \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile Name="$PROFILE_NAME" \
    --instance-initiated-shutdown-behavior terminate \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_PREFIX}]" \
    --query 'Instances[].InstanceId' --output text)
  RC=$?
  if [[ $RC -ne 0 ]]; then
    echo "On-demand launch failed. Consider using --count 1 or spot." >&2
    exit 1
  fi
fi
set -e
INSTANCE_IDS=( $RUN_OUT )

# Wait until running
AWS ec2 wait instance-running --instance-ids "${INSTANCE_IDS[@]}"

# Fetch IPs
AWS ec2 describe-instances --instance-ids "${INSTANCE_IDS[@]}" \
  --query 'Reservations[].Instances[].{Public:PublicIpAddress,Private:PrivateIpAddress,Id:InstanceId}' \
  --output text > cluster_hosts.txt

echo "\nCluster hosts written to cluster_hosts.txt (public_ip private_ip instance_id):"
cat cluster_hosts.txt

echo "\nNext: copy the repo and bootstrap on both nodes:"
echo "  scripts/aws/copy_repo.sh --region $REGION --key $KEY_NAME.pem --hosts cluster_hosts.txt"
