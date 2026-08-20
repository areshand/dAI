#!/usr/bin/env bash
set -euo pipefail

DAI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DAI_TOFU_DIR="$DAI_ROOT/infra/aws"
DAI_PROFILE=${DAI_AWS_PROFILE:-mi:scratchpad}
DAI_REGION=${DAI_AWS_REGION:-us-west-2}
DAI_TTL_MINUTES=${DAI_AWS_TTL_MINUTES:-180}
DAI_RUN_ID=${DAI_RUN_ID:-dai-$(date -u +%Y%m%d-%H%M%S)}
DAI_RESULT_DIR="$DAI_ROOT/prototype/results/aws"
DAI_TFVARS="$DAI_TOFU_DIR/run.auto.tfvars.json"
DAI_PLAN=$(mktemp /private/tmp/dai-aws-plan.XXXXXX)
DAI_COMMAND_INPUT=$(mktemp /private/tmp/dai-ssm-command.XXXXXX.json)
DAI_ENCODED_RESULT=$(mktemp /private/tmp/dai-result.XXXXXX.b64)
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
    echo "ERROR: run-scoped cost resources remain: instances=[$DAI_LIVE_INSTANCES] volumes=[$DAI_LIVE_VOLUMES]" >&2
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

if tofu -chdir="$DAI_TOFU_DIR" state list 2>/dev/null | grep -q .; then
  echo "ERROR: infra/aws already has managed resources; destroy or recover that run first." >&2
  exit 1
fi

DAI_ACCOUNT=$(aws sts get-caller-identity --profile "$DAI_PROFILE" --query Account --output text)
if [[ -z "$DAI_ACCOUNT" || "$DAI_ACCOUNT" == "None" ]]; then
  echo "ERROR: AWS profile $DAI_PROFILE did not resolve an account." >&2
  exit 1
fi

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
  --arg profile "$DAI_PROFILE" \
  --arg region "$DAI_REGION" \
  --arg run_id "$DAI_RUN_ID" \
  --arg expires "$DAI_EXPIRES_AT" \
  --arg near_subnet "$DAI_NEAR_SUBNET" \
  --arg far_subnet "$DAI_FAR_SUBNET" \
  --argjson ttl "$DAI_TTL_MINUTES" \
  '{aws_profile:$profile,aws_region:$region,run_id:$run_id,expires_at_utc:$expires,ttl_minutes:$ttl,near_subnet_id:$near_subnet,far_subnet_id:$far_subnet}' \
  > "$DAI_TFVARS"

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

echo "Waiting for all three nodes to become SSM-online"
DAI_SSM_ONLINE=false
for _ in $(seq 1 72); do
  # JMESPath uses backticks as literals; this is intentionally single-quoted.
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
  aws ssm wait command-executed --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --command-id "$DAI_TIMER_COMMAND" --instance-id "$DAI_TIMER_INSTANCE"
  DAI_TIMER_STATUS=$(aws ssm get-command-invocation --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --command-id "$DAI_TIMER_COMMAND" --instance-id "$DAI_TIMER_INSTANCE" --query StandardOutputContent --output text)
  if [[ "$DAI_TIMER_STATUS" != "active" ]]; then
    echo "ERROR: TTL termination timer is not active on $DAI_TIMER_INSTANCE." >&2
    exit 1
  fi
done
echo "Verified: independent TTL termination timer is active on every node"

DAI_REMOTE_COMMAND="set -euo pipefail; python3 /opt/dai/multi_node_moe.py --worker near=${DAI_NEAR_IP}:50123 --worker far=${DAI_FAR_IP}:50123 --requests 400 --payload-bytes 65536 --concurrency 1 --probe-samples 50 --blocks 10 --connect-timeout 120 --output /opt/dai/results/${DAI_RUN_ID}.json >&2; gzip -c /opt/dai/results/${DAI_RUN_ID}.json | base64"
jq -n \
  --arg instance "$DAI_COORDINATOR" \
  --arg command "$DAI_REMOTE_COMMAND" \
  --arg bucket "$DAI_BUCKET" \
  '{InstanceIds:[$instance],DocumentName:"AWS-RunShellScript",Parameters:{commands:[$command]},OutputS3BucketName:$bucket,OutputS3KeyPrefix:"ssm-output"}' \
  > "$DAI_COMMAND_INPUT"

DAI_COMMAND_ID=$(aws ssm send-command --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --cli-input-json "file://$DAI_COMMAND_INPUT" --query Command.CommandId --output text)
echo "Running counterbalanced placement evaluation through SSM command $DAI_COMMAND_ID"
if ! aws ssm wait command-executed --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --command-id "$DAI_COMMAND_ID" --instance-id "$DAI_COORDINATOR"; then
  aws ssm get-command-invocation --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --command-id "$DAI_COMMAND_ID" --instance-id "$DAI_COORDINATOR" --output json || true
  exit 1
fi

DAI_COMMAND_STATUS=$(aws ssm get-command-invocation --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --command-id "$DAI_COMMAND_ID" --instance-id "$DAI_COORDINATOR" --query Status --output text)
if [[ "$DAI_COMMAND_STATUS" != "Success" ]]; then
  echo "ERROR: experiment command status was $DAI_COMMAND_STATUS" >&2
  exit 1
fi

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
# Reading from stdin works with both GNU coreutils (`-d`) and macOS base64;
# macOS does not accept the input filename as a positional argument.
base64 -d < "$DAI_ENCODED_RESULT" | gzip -dc > "$DAI_RESULT_PATH"
jq -e '.schema == "multi-node-synthetic-moe.v1" and .correctness.outputs_identical_across_all_placements == true' \
  "$DAI_RESULT_PATH" >/dev/null

echo "Result captured: $DAI_RESULT_PATH"
jq '{discovery,placements,correctness,aggregate}' "$DAI_RESULT_PATH"
