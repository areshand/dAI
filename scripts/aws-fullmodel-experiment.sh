#!/usr/bin/env bash
set -euo pipefail

DAI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DAI_TOFU_DIR="$DAI_ROOT/infra/aws-fullmodel"
DAI_PROFILE=${DAI_AWS_PROFILE:-mi:scratchpad}
DAI_REGION=${DAI_AWS_REGION:-us-west-2}
DAI_MODEL_BUCKET=${DAI_MODEL_BUCKET:-}
DAI_MODEL_PREFIX=${DAI_MODEL_PREFIX:-qwen3-30b-a3b}
DAI_MODEL_LOCAL=${DAI_MODEL_LOCAL:-$DAI_ROOT/model-cache/qwen3-30b-a3b}
DAI_EXPERT_KEY=${DAI_EXPERT_KEY:-artifacts/qwen3-layer0-expert53.safetensors}
DAI_TTL_MINUTES=${DAI_AWS_TTL_MINUTES:-240}
DAI_RUN_ID=${DAI_RUN_ID:-dai-full-$(date -u +%Y%m%d-%H%M%S)}
DAI_RESULT_DIR="$DAI_ROOT/prototype/results/aws-fullmodel"
DAI_TFVARS="$DAI_TOFU_DIR/run.auto.tfvars.json"
DAI_PLAN=$(mktemp /private/tmp/dai-full-plan.XXXXXX)
DAI_COMMAND_INPUT=$(mktemp /private/tmp/dai-full-command.XXXXXX.json)
DAI_ENCODED_RESULT=$(mktemp /private/tmp/dai-full-result.XXXXXX.b64)
DAI_CLEANUP_FAILED=0

cleanup() {
  DAI_EXIT_STATUS=$?
  trap - EXIT INT TERM
  if [[ -f "$DAI_TFVARS" ]]; then
    echo "Destroying run-scoped AWS resources for $DAI_RUN_ID"
    if ! tofu -chdir="$DAI_TOFU_DIR" destroy -auto-approve; then
      DAI_CLEANUP_FAILED=1
    fi
  fi

  DAI_LIVE_INSTANCES=$(aws ec2 describe-instances \
    --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --filters "Name=tag:Project,Values=dAI" "Name=tag:RunId,Values=$DAI_RUN_ID" \
      "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
    --query 'Reservations[].Instances[].InstanceId' --output text || true)
  DAI_LIVE_VOLUMES=$(aws ec2 describe-volumes \
    --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --filters "Name=tag:Project,Values=dAI" "Name=tag:RunId,Values=$DAI_RUN_ID" \
    --query 'Volumes[].VolumeId' --output text || true)
  if [[ -n "$DAI_LIVE_INSTANCES" || -n "$DAI_LIVE_VOLUMES" ]]; then
    echo "ERROR: cost resources remain: instances=[$DAI_LIVE_INSTANCES] volumes=[$DAI_LIVE_VOLUMES]" >&2
    DAI_CLEANUP_FAILED=1
  else
    echo "Verified: no live instances or EBS volumes remain for $DAI_RUN_ID"
  fi

  rm -f "$DAI_PLAN" "$DAI_COMMAND_INPUT" "$DAI_ENCODED_RESULT" "$DAI_TFVARS"
  if [[ "$DAI_CLEANUP_FAILED" -ne 0 ]]; then
    exit 1
  fi
  exit "$DAI_EXIT_STATUS"
}
trap cleanup EXIT INT TERM

wait_for_command() {
  local command_id=$1
  local instance_id=$2
  local attempts=$3
  local status
  for _ in $(seq 1 "$attempts"); do
    status=$(aws ssm get-command-invocation --profile "$DAI_PROFILE" --region "$DAI_REGION" \
      --command-id "$command_id" --instance-id "$instance_id" --query Status --output text 2>/dev/null || true)
    case "$status" in
      Success) return 0 ;;
      Failed|Cancelled|TimedOut|Cancelling)
        aws ssm get-command-invocation --profile "$DAI_PROFILE" --region "$DAI_REGION" \
          --command-id "$command_id" --instance-id "$instance_id" --output json || true
        return 1
        ;;
    esac
    sleep 10
  done
  echo "ERROR: SSM command $command_id did not finish on $instance_id." >&2
  return 1
}

if tofu -chdir="$DAI_TOFU_DIR" state list 2>/dev/null | grep -q .; then
  echo "ERROR: infra/aws-fullmodel already has managed resources." >&2
  exit 1
fi
if [[ ! -s "$DAI_MODEL_LOCAL/model.safetensors.index.json" ]]; then
  echo "ERROR: local pinned model is missing at $DAI_MODEL_LOCAL." >&2
  exit 1
fi

DAI_ACCOUNT=$(aws sts get-caller-identity --profile "$DAI_PROFILE" --query Account --output text)
DAI_MODEL_BUCKET=${DAI_MODEL_BUCKET:-dai-${DAI_ACCOUNT}-model-cache-${DAI_REGION}}

echo "Verifying that the protected S3 model cache exactly covers the local checkpoint"
DAI_SYNC_DIFF=$(aws s3 sync "$DAI_MODEL_LOCAL" "s3://$DAI_MODEL_BUCKET/$DAI_MODEL_PREFIX/" \
  --profile "$DAI_PROFILE" --region "$DAI_REGION" --exclude '.cache/*' --delete --dryrun)
if [[ -n "$DAI_SYNC_DIFF" ]]; then
  echo "ERROR: S3 model cache is incomplete or differs from the local checkpoint:" >&2
  echo "$DAI_SYNC_DIFF" >&2
  exit 1
fi
aws s3api head-object --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --bucket "$DAI_MODEL_BUCKET" --key "$DAI_EXPERT_KEY" >/dev/null

DAI_NEAR_SUBNET=$(aws ec2 describe-subnets --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --filters Name=default-for-az,Values=true Name=availability-zone,Values="${DAI_REGION}a" \
  --query 'Subnets[0].SubnetId' --output text)
DAI_FAR_SUBNET=$(aws ec2 describe-subnets --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --filters Name=default-for-az,Values=true Name=availability-zone,Values="${DAI_REGION}c" \
  --query 'Subnets[0].SubnetId' --output text)
if [[ "$DAI_NEAR_SUBNET" == "None" || "$DAI_FAR_SUBNET" == "None" ]]; then
  echo "ERROR: default public subnets were not found in ${DAI_REGION}a and ${DAI_REGION}c." >&2
  exit 1
fi

DAI_EXPIRES_AT=$(python3 -c 'import datetime,sys; print((datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=int(sys.argv[1]))).replace(microsecond=0).isoformat().replace("+00:00","Z"))' "$DAI_TTL_MINUTES")
mkdir -p "$DAI_RESULT_DIR"
jq -n \
  --arg profile "$DAI_PROFILE" --arg region "$DAI_REGION" --arg run_id "$DAI_RUN_ID" \
  --arg expires "$DAI_EXPIRES_AT" --arg near_subnet "$DAI_NEAR_SUBNET" \
  --arg far_subnet "$DAI_FAR_SUBNET" --arg model_bucket "$DAI_MODEL_BUCKET" \
  --arg model_prefix "$DAI_MODEL_PREFIX" --arg expert_key "$DAI_EXPERT_KEY" \
  --argjson ttl "$DAI_TTL_MINUTES" \
  '{aws_profile:$profile,aws_region:$region,run_id:$run_id,expires_at_utc:$expires,
    ttl_minutes:$ttl,near_subnet_id:$near_subnet,far_subnet_id:$far_subnet,
    model_bucket:$model_bucket,model_prefix:$model_prefix,expert_key:$expert_key}' > "$DAI_TFVARS"

echo "Planning $DAI_RUN_ID in account $DAI_ACCOUNT ($DAI_REGION), hard expiry $DAI_EXPIRES_AT"
tofu -chdir="$DAI_TOFU_DIR" init -input=false
tofu -chdir="$DAI_TOFU_DIR" plan -input=false -out="$DAI_PLAN"
tofu -chdir="$DAI_TOFU_DIR" apply -input=false "$DAI_PLAN"

DAI_COORDINATOR=$(tofu -chdir="$DAI_TOFU_DIR" output -raw coordinator_instance_id)
DAI_NEAR_INSTANCE=$(tofu -chdir="$DAI_TOFU_DIR" output -raw near_instance_id)
DAI_FAR_INSTANCE=$(tofu -chdir="$DAI_TOFU_DIR" output -raw far_instance_id)
DAI_NEAR_IP=$(tofu -chdir="$DAI_TOFU_DIR" output -raw near_private_ip)
DAI_FAR_IP=$(tofu -chdir="$DAI_TOFU_DIR" output -raw far_private_ip)
DAI_BUCKET=$(tofu -chdir="$DAI_TOFU_DIR" output -raw result_bucket)

echo "Waiting for all three full-model nodes to become SSM-online"
DAI_SSM_ONLINE=false
for _ in $(seq 1 72); do
  # shellcheck disable=SC2016
  DAI_ONLINE_COUNT=$(aws ssm describe-instance-information \
    --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --filters "Key=InstanceIds,Values=$DAI_COORDINATOR,$DAI_NEAR_INSTANCE,$DAI_FAR_INSTANCE" \
    --query 'length(InstanceInformationList[?PingStatus==`Online`])' --output text 2>/dev/null || true)
  if [[ "$DAI_ONLINE_COUNT" == "3" ]]; then
    DAI_SSM_ONLINE=true
    break
  fi
  sleep 5
done
if [[ "$DAI_SSM_ONLINE" != "true" ]]; then
  echo "ERROR: all three nodes did not become SSM-online." >&2
  exit 1
fi

DAI_TIMER_COMMAND=$(aws ssm send-command --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --instance-ids "$DAI_COORDINATOR" "$DAI_NEAR_INSTANCE" "$DAI_FAR_INSTANCE" \
  --document-name AWS-RunShellScript --parameters 'commands=["systemctl is-active dai-ttl-terminate.timer"]' \
  --query Command.CommandId --output text)
for DAI_TIMER_INSTANCE in "$DAI_COORDINATOR" "$DAI_NEAR_INSTANCE" "$DAI_FAR_INSTANCE"; do
  wait_for_command "$DAI_TIMER_COMMAND" "$DAI_TIMER_INSTANCE" 30
  DAI_TIMER_STATUS=$(aws ssm get-command-invocation --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --command-id "$DAI_TIMER_COMMAND" --instance-id "$DAI_TIMER_INSTANCE" \
    --query StandardOutputContent --output text)
  if [[ "$DAI_TIMER_STATUS" != "active" ]]; then
    echo "ERROR: TTL termination timer is not active on $DAI_TIMER_INSTANCE." >&2
    exit 1
  fi
done
echo "Verified: independent TTL termination timer is active on every node"

# The command substitution is intentionally passed literally for execution on EC2.
# shellcheck disable=SC2016
DAI_COORD_READY=$(aws ssm send-command --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --instance-ids "$DAI_COORDINATOR" --document-name AWS-RunShellScript --timeout-seconds 2400 \
  --parameters 'commands=["for i in $(seq 1 180); do test -f /opt/dai/ready && test -s /opt/dai/model/model-00016-of-00016.safetensors && exit 0; sleep 10; done; exit 1"]' \
  --query Command.CommandId --output text)
# shellcheck disable=SC2016
DAI_WORKERS_READY=$(aws ssm send-command --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --instance-ids "$DAI_NEAR_INSTANCE" "$DAI_FAR_INSTANCE" --document-name AWS-RunShellScript --timeout-seconds 2400 \
  --parameters 'commands=["for i in $(seq 1 180); do test -f /opt/dai/ready && systemctl is-active --quiet dai-real-expert.service && exit 0; sleep 10; done; exit 1"]' \
  --query Command.CommandId --output text)
echo "Waiting for dependency installation and S3 weight synchronization"
wait_for_command "$DAI_COORD_READY" "$DAI_COORDINATOR" 240
wait_for_command "$DAI_WORKERS_READY" "$DAI_NEAR_INSTANCE" 240
wait_for_command "$DAI_WORKERS_READY" "$DAI_FAR_INSTANCE" 240
echo "Verified: model and both real-expert services are ready"

DAI_REMOTE_COMMAND="set -euo pipefail; /opt/dai/venv/bin/python /opt/dai/full_model_multi_worker_eval.py --model-path /opt/dai/model --worker near=${DAI_NEAR_IP}:50126 --worker far=${DAI_FAR_IP}:50126 --device cpu --layer 0 --expert 53 --blocks 5 --probe-samples 20 --timeout 120 --output /opt/dai/results/${DAI_RUN_ID}.json >&2; gzip -c /opt/dai/results/${DAI_RUN_ID}.json | base64"
jq -n --arg instance "$DAI_COORDINATOR" --arg command "$DAI_REMOTE_COMMAND" --arg bucket "$DAI_BUCKET" \
  '{InstanceIds:[$instance],DocumentName:"AWS-RunShellScript",TimeoutSeconds:7200,
    Parameters:{commands:[$command]},OutputS3BucketName:$bucket,OutputS3KeyPrefix:"ssm-output"}' \
  > "$DAI_COMMAND_INPUT"
DAI_COMMAND_ID=$(aws ssm send-command --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --cli-input-json "file://$DAI_COMMAND_INPUT" --query Command.CommandId --output text)
echo "Running real full-model placement evaluation through SSM command $DAI_COMMAND_ID"
wait_for_command "$DAI_COMMAND_ID" "$DAI_COORDINATOR" 720

DAI_STDOUT_KEY=$(aws s3api list-objects-v2 --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --bucket "$DAI_BUCKET" --prefix "ssm-output/$DAI_COMMAND_ID" \
  --query "Contents[?ends_with(Key, 'stdout')].Key | [0]" --output text)
if [[ -z "$DAI_STDOUT_KEY" || "$DAI_STDOUT_KEY" == "None" ]]; then
  echo "ERROR: SSM did not publish the encoded result object." >&2
  exit 1
fi
aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  "s3://$DAI_BUCKET/$DAI_STDOUT_KEY" "$DAI_ENCODED_RESULT"
DAI_RESULT_PATH="$DAI_RESULT_DIR/${DAI_RUN_ID}.json"
base64 -d < "$DAI_ENCODED_RESULT" | gzip -dc > "$DAI_RESULT_PATH"
jq -e '.schema == "qwen3-full-model-multi-worker-placement.v1" and
  .summary.all_routes_equal == true and .summary.all_logits_allclose == true and
  .summary.all_tokens_equal == true' "$DAI_RESULT_PATH" >/dev/null

echo "Result captured: $DAI_RESULT_PATH"
jq '{latency_aware_worker,discovery,summary}' "$DAI_RESULT_PATH"
