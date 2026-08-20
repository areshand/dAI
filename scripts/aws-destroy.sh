#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[a-z0-9][a-z0-9-]{5,48}$ ]]; then
  echo "usage: $0 <exact-run-id>" >&2
  exit 2
fi

DAI_RUN_ID=$1
DAI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DAI_TOFU_DIR="$DAI_ROOT/infra/aws"
DAI_PROFILE=${DAI_AWS_PROFILE:-mi:scratchpad}
DAI_REGION=${DAI_AWS_REGION:-us-west-2}

if [[ -f "$DAI_TOFU_DIR/run.auto.tfvars.json" ]] && \
   [[ $(jq -r .run_id "$DAI_TOFU_DIR/run.auto.tfvars.json") == "$DAI_RUN_ID" ]]; then
  tofu -chdir="$DAI_TOFU_DIR" destroy -auto-approve
else
  DAI_INSTANCE_IDS=$(aws ec2 describe-instances --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --filters "Name=tag:Project,Values=dAI" "Name=tag:RunId,Values=$DAI_RUN_ID" \
      Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  if [[ -n "$DAI_INSTANCE_IDS" ]]; then
    read -r -a DAI_INSTANCE_ARRAY <<< "$DAI_INSTANCE_IDS"
    aws ec2 terminate-instances --profile "$DAI_PROFILE" --region "$DAI_REGION" \
      --instance-ids "${DAI_INSTANCE_ARRAY[@]}"
    aws ec2 wait instance-terminated --profile "$DAI_PROFILE" --region "$DAI_REGION" \
      --instance-ids "${DAI_INSTANCE_ARRAY[@]}"
  fi
  echo "Emergency cleanup terminated exact tagged instances; use the retained OpenTofu state to remove non-compute resources." >&2
fi

DAI_REMAINING=$(aws ec2 describe-instances --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --filters "Name=tag:Project,Values=dAI" "Name=tag:RunId,Values=$DAI_RUN_ID" \
    Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [[ -n "$DAI_REMAINING" ]]; then
  echo "ERROR: instances still remain for $DAI_RUN_ID: $DAI_REMAINING" >&2
  exit 1
fi
echo "No live instances remain for $DAI_RUN_ID"
