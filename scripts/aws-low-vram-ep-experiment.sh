#!/usr/bin/env bash
set -euo pipefail

DAI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DAI_TOFU_DIR="$DAI_ROOT/infra/aws-low-vram-ep"
DAI_PROFILE=${DAI_AWS_PROFILE:-mi:scratchpad}
DAI_REGION=${DAI_AWS_REGION:-us-west-2}
DAI_AZ=${DAI_AWS_AVAILABILITY_ZONE:-us-west-2a}
DAI_INSTANCE_TYPE=${DAI_AWS_GPU_INSTANCE_TYPE:-gr6.4xlarge}
DAI_WORKERS=4
DAI_TTL_MINUTES=${DAI_AWS_TTL_MINUTES:-180}
DAI_SGLANG_IMAGE=${DAI_SGLANG_IMAGE:-lmsysorg/sglang:v0.5.16}
DAI_MODEL_PREFIX=${DAI_MODEL_PREFIX:-qwen3-30b-a3b}
DAI_MODEL_LOCAL=${DAI_MODEL_LOCAL:-$DAI_ROOT/model-cache/qwen3-30b-a3b}
DAI_MODEL_BUCKET=${DAI_MODEL_BUCKET:-}
DAI_MEM_FRACTION_STATIC=${DAI_MEM_FRACTION_STATIC:-0.90}
DAI_EXPERT_PLACEMENT_FILE=${DAI_EXPERT_PLACEMENT_FILE:-}
DAI_EXPERT_PLACEMENT_REPORT=${DAI_EXPERT_PLACEMENT_REPORT:-}
DAI_CAPTURE_ROUTING=${DAI_CAPTURE_ROUTING:-1}
DAI_RUN_ID=${DAI_RUN_ID:-dai-ep4-$(date -u +%Y%m%d-%H%M%S)}
DAI_RESULT_DIR="$DAI_ROOT/prototype/results/aws-low-vram-ep/$DAI_RUN_ID"
DAI_TFVARS="$DAI_TOFU_DIR/run.auto.tfvars.json"
DAI_PLAN=$(mktemp /private/tmp/dai-low-vram-ep-plan.XXXXXX)
DAI_COMMAND_INPUT=$(mktemp /private/tmp/dai-low-vram-ep-command.XXXXXX)
DAI_CLEANUP_FAILED=0
DAI_BUCKET=""
DAI_INSTANCE_IDS=()
DAI_PRIVATE_IPS=()

cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  if [[ -f "$DAI_TFVARS" ]]; then
    echo "Destroying run-scoped AWS resources for $DAI_RUN_ID"
    if ! tofu -chdir="$DAI_TOFU_DIR" destroy -auto-approve; then
      DAI_CLEANUP_FAILED=1
    fi
  fi

  local live_instances
  local live_volumes
  live_instances=$(aws ec2 describe-instances --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --filters "Name=tag:Project,Values=dAI" "Name=tag:RunId,Values=$DAI_RUN_ID" \
      "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
    --query 'Reservations[].Instances[].InstanceId' --output text || true)
  live_volumes=$(aws ec2 describe-volumes --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --filters "Name=tag:Project,Values=dAI" "Name=tag:RunId,Values=$DAI_RUN_ID" \
    --query 'Volumes[].VolumeId' --output text || true)
  if [[ -n "$live_instances" || -n "$live_volumes" ]]; then
    echo "ERROR: cost resources remain: instances=[$live_instances] volumes=[$live_volumes]" >&2
    DAI_CLEANUP_FAILED=1
  else
    echo "Verified: no live instances or EBS volumes remain for $DAI_RUN_ID"
  fi

  rm -f "$DAI_PLAN" "$DAI_COMMAND_INPUT" "$DAI_TFVARS"
  if [[ "$DAI_CLEANUP_FAILED" -ne 0 ]]; then
    exit 1
  fi
  exit "$exit_status"
}
trap cleanup EXIT INT TERM

send_command() {
  local instance_id=$1
  local command=$2
  local timeout_seconds=$3
  local encoded_command
  local bash_command
  encoded_command=$(printf '%s' "$command" | base64 | tr -d '\n')
  bash_command="echo '$encoded_command' | base64 -d | bash"
  jq -n --arg instance "$instance_id" --arg command "$bash_command" --arg bucket "$DAI_BUCKET" \
    --argjson timeout "$timeout_seconds" \
    '{InstanceIds:[$instance],DocumentName:"AWS-RunShellScript",TimeoutSeconds:$timeout,
      Parameters:{commands:[$command]},OutputS3BucketName:$bucket,OutputS3KeyPrefix:"ssm-output"}' \
    > "$DAI_COMMAND_INPUT"
  aws ssm send-command --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --cli-input-json "file://$DAI_COMMAND_INPUT" --query Command.CommandId --output text
}

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

collect_rank_artifacts() {
  local phase=$1
  local command_ids=()
  local rank
  local command
  for rank in $(seq 0 $((DAI_WORKERS - 1))); do
    command="set -euo pipefail; rank=\$(cat /opt/dai/node-rank); mkdir -p /opt/dai/results; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader > /opt/dai/results/rank-\${rank}-${phase}-gpu.csv; ip -s -j link show > /opt/dai/results/rank-\${rank}-${phase}-network.json; docker inspect dai-sglang > /opt/dai/results/rank-\${rank}-${phase}-container.json 2>/dev/null || true; docker logs dai-sglang > /opt/dai/results/rank-\${rank}-${phase}-server.log 2>&1 || true; aws s3 cp /opt/dai/results/rank-\${rank}-${phase}-gpu.csv s3://$DAI_BUCKET/results/ --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/rank-\${rank}-${phase}-network.json s3://$DAI_BUCKET/results/ --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/rank-\${rank}-${phase}-container.json s3://$DAI_BUCKET/results/ --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/rank-\${rank}-${phase}-server.log s3://$DAI_BUCKET/results/ --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/gpu-contract.csv s3://$DAI_BUCKET/results/rank-\${rank}-gpu-contract.csv --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/sglang-image.json s3://$DAI_BUCKET/results/rank-\${rank}-sglang-image.json --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/qwen3-moe-patch-sha256.txt s3://$DAI_BUCKET/results/rank-\${rank}-qwen3-moe-patch-sha256.txt --region $DAI_REGION --only-show-errors"
    command_ids+=("$(send_command "${DAI_INSTANCE_IDS[$rank]}" "$command" 600)")
  done
  for rank in $(seq 0 $((DAI_WORKERS - 1))); do
    wait_for_command "${command_ids[$rank]}" "${DAI_INSTANCE_IDS[$rank]}" 70
  done
}

download_artifacts() {
  mkdir -p "$DAI_RESULT_DIR"
  aws s3 sync "s3://$DAI_BUCKET/results/" "$DAI_RESULT_DIR/" \
    --profile "$DAI_PROFILE" --region "$DAI_REGION" --only-show-errors
}

launch_sglang() {
  local phase=$1
  local extra_args=${2:-}
  local init_expert_location=${3:-trivial}
  local command_ids=()
  local rank
  local launch_command
  echo "Launching $phase TP4/DP4/EP4 server"
  for rank in $(seq 0 $((DAI_WORKERS - 1))); do
    launch_command="set -euo pipefail; docker rm -f dai-sglang >/dev/null 2>&1 || true; iface=\$(ip route show default | awk '{print \$5; exit}'); docker run -d --name dai-sglang --gpus all --ipc=host --network host --shm-size 8g -e NCCL_SOCKET_IFNAME=\$iface -e GLOO_SOCKET_IFNAME=\$iface -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=INFO -v /opt/dai/model:/models/qwen3:ro -v /opt/dai:/opt/dai -v /opt/dai/qwen3_moe.py:/sgl-workspace/sglang/python/sglang/srt/models/qwen3_moe.py:ro '$DAI_SGLANG_IMAGE' python3 -m sglang.launch_server --model-path /models/qwen3 --served-model-name qwen3-30b-a3b --host 0.0.0.0 --port 30000 --dist-init-addr '$DAI_HEAD_IP:20000' --nnodes '$DAI_WORKERS' --node-rank '$rank' --tp-size '$DAI_WORKERS' --dp-size '$DAI_WORKERS' --ep-size '$DAI_WORKERS' --enable-dp-attention --init-expert-location '$init_expert_location' --dtype bfloat16 --context-length 2048 --max-running-requests '$DAI_WORKERS' --mem-fraction-static '$DAI_MEM_FRACTION_STATIC' --disable-cuda-graph --disable-custom-all-reduce --random-seed 1234 --watchdog-timeout 900 $extra_args"
    command_ids+=("$(send_command "${DAI_INSTANCE_IDS[$rank]}" "$launch_command" 600)")
  done
  for rank in $(seq 0 $((DAI_WORKERS - 1))); do
    wait_for_command "${command_ids[$rank]}" "${DAI_INSTANCE_IDS[$rank]}" 70
  done
}

wait_for_sglang_health() {
  local phase=$1
  local health_command
  local command_id
  health_command="set -euo pipefail; for i in \$(seq 1 240); do if curl -fsS http://127.0.0.1:30000/health >/dev/null; then exit 0; fi; if ! docker inspect -f '{{.State.Running}}' dai-sglang 2>/dev/null | grep -q true; then docker logs dai-sglang; exit 1; fi; sleep 5; done; docker logs dai-sglang; exit 1"
  command_id=$(send_command "${DAI_INSTANCE_IDS[0]}" "$health_command" 1500)
  echo "Waiting for $phase distributed model initialization ($command_id)"
  wait_for_command "$command_id" "${DAI_INSTANCE_IDS[0]}" 160
}

collect_routing_artifacts() {
  local command_ids=()
  local rank
  local command
  for rank in $(seq 0 $((DAI_WORKERS - 1))); do
    command="set -euo pipefail; rank=\$(cat /opt/dai/node-rank); docker logs dai-sglang > /opt/dai/results/rank-\${rank}-routing-server.log 2>&1 || true; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader > /opt/dai/results/rank-\${rank}-routing-gpu.csv; aws s3 cp /opt/dai/results/rank-\${rank}-routing-server.log s3://$DAI_BUCKET/results/ --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/rank-\${rank}-routing-gpu.csv s3://$DAI_BUCKET/results/ --region $DAI_REGION --only-show-errors"
    command_ids+=("$(send_command "${DAI_INSTANCE_IDS[$rank]}" "$command" 600)")
  done
  for rank in $(seq 0 $((DAI_WORKERS - 1))); do
    wait_for_command "${command_ids[$rank]}" "${DAI_INSTANCE_IDS[$rank]}" 70
  done
}

if tofu -chdir="$DAI_TOFU_DIR" state list 2>/dev/null | grep -q .; then
  echo "ERROR: infra/aws-low-vram-ep already has managed resources." >&2
  exit 1
fi
if [[ ! -s "$DAI_MODEL_LOCAL/model.safetensors.index.json" ]]; then
  echo "ERROR: local pinned model is missing at $DAI_MODEL_LOCAL." >&2
  exit 1
fi
if [[ "$DAI_CAPTURE_ROUTING" != "0" && "$DAI_CAPTURE_ROUTING" != "1" ]]; then
  echo "ERROR: DAI_CAPTURE_ROUTING must be 0 or 1." >&2
  exit 1
fi
if [[ -n "$DAI_EXPERT_PLACEMENT_FILE" ]]; then
  test -s "$DAI_EXPERT_PLACEMENT_FILE"
  test -s "$DAI_EXPERT_PLACEMENT_REPORT"
  jq -e '.physical_to_logical_map | length == 48 and
    all(.[]; length == 128 and (sort == [range(0;128)]))' "$DAI_EXPERT_PLACEMENT_FILE" >/dev/null
fi

DAI_ACCOUNT=$(aws sts get-caller-identity --profile "$DAI_PROFILE" --query Account --output text)
DAI_MODEL_BUCKET=${DAI_MODEL_BUCKET:-dai-${DAI_ACCOUNT}-model-cache-${DAI_REGION}}
echo "Verifying the protected S3 cache against the pinned local checkpoint"
DAI_SYNC_DIFF=$(aws s3 sync "$DAI_MODEL_LOCAL" "s3://$DAI_MODEL_BUCKET/$DAI_MODEL_PREFIX/" \
  --profile "$DAI_PROFILE" --region "$DAI_REGION" --exclude '.cache/*' --delete --dryrun)
if [[ -n "$DAI_SYNC_DIFF" ]]; then
  echo "ERROR: S3 model cache is incomplete or differs from the local checkpoint." >&2
  echo "$DAI_SYNC_DIFF" >&2
  exit 1
fi

case "$DAI_INSTANCE_TYPE" in
  g6.4xlarge) DAI_NODE_HOURLY_USD=1.3232 ;;
  gr6.4xlarge) DAI_NODE_HOURLY_USD=1.5392 ;;
esac
DAI_TOTAL_HOURLY_USD=$(awk -v node="$DAI_NODE_HOURLY_USD" -v workers="$DAI_WORKERS" 'BEGIN { printf "%.3f", node * workers }')
if ! awk -v total="$DAI_TOTAL_HOURLY_USD" 'BEGIN { exit !(total <= 8.60) }'; then
  echo "ERROR: planned hourly compute cost is above the pre-registered $8.60 ceiling." >&2
  exit 1
fi

DAI_SUBNET=$(aws ec2 describe-subnets --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --filters Name=default-for-az,Values=true "Name=availability-zone,Values=$DAI_AZ" \
  --query 'Subnets[0].SubnetId' --output text)
if [[ "$DAI_SUBNET" == "None" || -z "$DAI_SUBNET" ]]; then
  echo "ERROR: no default subnet exists in $DAI_AZ." >&2
  exit 1
fi

DAI_EXPIRES_AT=$(python3 -c 'import datetime,sys; print((datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=int(sys.argv[1]))).replace(microsecond=0).isoformat().replace("+00:00","Z"))' "$DAI_TTL_MINUTES")
mkdir -p "$DAI_RESULT_DIR"
jq -n --arg profile "$DAI_PROFILE" --arg region "$DAI_REGION" --arg run_id "$DAI_RUN_ID" \
  --arg expires "$DAI_EXPIRES_AT" --arg subnet "$DAI_SUBNET" --arg model_bucket "$DAI_MODEL_BUCKET" \
  --arg model_prefix "$DAI_MODEL_PREFIX" --arg instance_type "$DAI_INSTANCE_TYPE" \
  --arg sglang_image "$DAI_SGLANG_IMAGE" --argjson ttl "$DAI_TTL_MINUTES" \
  --argjson worker_count "$DAI_WORKERS" \
  '{aws_profile:$profile,aws_region:$region,run_id:$run_id,expires_at_utc:$expires,
    ttl_minutes:$ttl,worker_count:$worker_count,subnet_id:$subnet,model_bucket:$model_bucket,
    model_prefix:$model_prefix,instance_type:$instance_type,sglang_image:$sglang_image}' > "$DAI_TFVARS"

jq -n --arg run_id "$DAI_RUN_ID" --arg region "$DAI_REGION" --arg az "$DAI_AZ" \
  --arg instance_type "$DAI_INSTANCE_TYPE" --arg image "$DAI_SGLANG_IMAGE" \
  --arg model_prefix "$DAI_MODEL_PREFIX" --arg expires "$DAI_EXPIRES_AT" \
  --arg placement_file "$DAI_EXPERT_PLACEMENT_FILE" --argjson capture_routing "$DAI_CAPTURE_ROUTING" \
  --argjson workers "$DAI_WORKERS" --argjson gpu_vram_mib 22888 --argjson gpu_vram_limit_mib 24576 \
  --arg hourly "$DAI_TOTAL_HOURLY_USD" \
  '{schema:"dai-low-vram-ep-manifest.v1",run_id:$run_id,region:$region,availability_zone:$az,
    instance_type:$instance_type,worker_count:$workers,gpu_vram_mib_per_worker:$gpu_vram_mib,
    gpu_vram_limit_mib_per_worker:$gpu_vram_limit_mib,total_gpu_vram_mib:($gpu_vram_mib*$workers),
    parallelism:{tp:4,dp:4,ep:4,attention_tp:1},
    model:$model_prefix,dtype:"bfloat16",runtime_image:$image,planned_hourly_compute_usd:($hourly|tonumber),
    workload:{batch:1,input_tokens:1000,output_tokens:256,cache_policy:"cold",max_running_requests:$workers},
    runtime_patch:{name:"qwen3-forward-normal-logical-to-physical-dispatch",
      pinned_source_sha256:"b18eb188c594c41ff58debe6df72cf852975b0504b5ae0513ccb4be75fea1bc2"},
    placement_comparison:{enabled:($placement_file != ""),
      execution_order:["trivial-pre","optimized","trivial-post"]},
    routing_trace:{separate_instrumented_phase:($capture_routing == 1),num_layers:48,experts_per_layer:128,
      experts_per_token:8,expert_parallel_size:4,placement:"trivial-contiguous",
      warmups:2,measured_requests:10,workload_scope:"fixed-benchmark-prompt"},
    expires_at_utc:$expires}' \
  > "$DAI_RESULT_DIR/manifest.json"

echo "Planning $DAI_RUN_ID: $DAI_WORKERS x $DAI_INSTANCE_TYPE at \$$DAI_TOTAL_HOURLY_USD/hour, hard expiry $DAI_EXPIRES_AT"
tofu -chdir="$DAI_TOFU_DIR" init -input=false
tofu -chdir="$DAI_TOFU_DIR" plan -input=false -out="$DAI_PLAN"
tofu -chdir="$DAI_TOFU_DIR" apply -input=false "$DAI_PLAN"

mapfile -t DAI_INSTANCE_IDS < <(tofu -chdir="$DAI_TOFU_DIR" output -json instance_ids | jq -r '.[]')
mapfile -t DAI_PRIVATE_IPS < <(tofu -chdir="$DAI_TOFU_DIR" output -json private_ips | jq -r '.[]')
DAI_BUCKET=$(tofu -chdir="$DAI_TOFU_DIR" output -raw result_bucket)
DAI_HEAD_IP=${DAI_PRIVATE_IPS[0]}

echo "Waiting for all four workers to become SSM-online and finish model/image setup"
for rank in $(seq 0 $((DAI_WORKERS - 1))); do
  DAI_PING=""
  for _ in $(seq 1 120); do
    DAI_PING=$(aws ssm describe-instance-information --profile "$DAI_PROFILE" --region "$DAI_REGION" \
      --filters "Key=InstanceIds,Values=${DAI_INSTANCE_IDS[$rank]}" \
      --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
    [[ "$DAI_PING" == "Online" ]] && break
    sleep 5
  done
  [[ "$DAI_PING" == "Online" ]] || { echo "ERROR: rank $rank did not become SSM-online." >&2; exit 1; }
done

DAI_READY_COMMANDS=()
for rank in $(seq 0 $((DAI_WORKERS - 1))); do
  # The command substitution is intentionally executed on EC2.
  # shellcheck disable=SC2016
  DAI_READY_COMMANDS+=("$(send_command "${DAI_INSTANCE_IDS[$rank]}" 'for i in $(seq 1 360); do test -f /opt/dai/ready && systemctl is-active --quiet dai-ttl-terminate.timer && exit 0; sleep 5; done; exit 1' 2100)")
done
for rank in $(seq 0 $((DAI_WORKERS - 1))); do
  wait_for_command "${DAI_READY_COMMANDS[$rank]}" "${DAI_INSTANCE_IDS[$rank]}" 220
done

if [[ -n "$DAI_EXPERT_PLACEMENT_FILE" ]]; then
  cp "$DAI_EXPERT_PLACEMENT_FILE" "$DAI_RESULT_DIR/optimized-expert-placement.json"
  cp "$DAI_EXPERT_PLACEMENT_REPORT" "$DAI_RESULT_DIR/expert-placement-optimization.json"
  aws s3 cp "$DAI_EXPERT_PLACEMENT_FILE" "s3://$DAI_BUCKET/control/optimized-expert-placement.json" \
    --profile "$DAI_PROFILE" --region "$DAI_REGION" --only-show-errors
  DAI_PLACEMENT_COMMANDS=()
  for rank in $(seq 0 $((DAI_WORKERS - 1))); do
    DAI_PLACEMENT_COMMANDS+=("$(send_command "${DAI_INSTANCE_IDS[$rank]}" \
      "aws s3 cp s3://$DAI_BUCKET/control/optimized-expert-placement.json /opt/dai/optimized-expert-placement.json --region $DAI_REGION --only-show-errors" 600)")
  done
  for rank in $(seq 0 $((DAI_WORKERS - 1))); do
    wait_for_command "${DAI_PLACEMENT_COMMANDS[$rank]}" "${DAI_INSTANCE_IDS[$rank]}" 70
  done
fi

launch_sglang performance
if ! wait_for_sglang_health performance; then
  collect_rank_artifacts startup-failed || true
  download_artifacts || true
  echo "ERROR: the four-rank BF16 server did not initialize; retained logs show whether the gate was VRAM or runtime compatibility." >&2
  exit 1
fi

collect_rank_artifacts before

smoke_command="set -euo pipefail; docker exec dai-sglang python3 /opt/dai/generation_benchmark.py --endpoint http://127.0.0.1:30000 --model qwen3-30b-a3b --tokenizer /models/qwen3 --variant low-vram-ep4-smoke --prompt-file /opt/dai/benchmark-prompt.md --prompt-tokens 1000 --max-tokens 32 --warmups 1 --repetitions 1 --cache-policy cold --nonce dai-generation-v2 --output /opt/dai/results/smoke.json; aws s3 cp /opt/dai/results/smoke.json s3://$DAI_BUCKET/results/smoke.json --region $DAI_REGION --only-show-errors"
DAI_SMOKE_COMMAND=$(send_command "${DAI_INSTANCE_IDS[0]}" "$smoke_command" 1800)
echo "Running one cold joint-inference smoke request ($DAI_SMOKE_COMMAND)"
wait_for_command "$DAI_SMOKE_COMMAND" "${DAI_INSTANCE_IDS[0]}" 190

benchmark_command="set -euo pipefail; docker exec dai-sglang python3 /opt/dai/generation_benchmark.py --endpoint http://127.0.0.1:30000 --model qwen3-30b-a3b --tokenizer /models/qwen3 --variant low-vram-ep4-bf16 --prompt-file /opt/dai/benchmark-prompt.md --prompt-tokens 1000 --max-tokens 256 --warmups 2 --repetitions 10 --cache-policy cold --nonce dai-generation-v2 --output /opt/dai/results/benchmark.json; curl -fsS http://127.0.0.1:30000/get_server_info > /opt/dai/results/server-info.json; aws s3 cp /opt/dai/results/benchmark.json s3://$DAI_BUCKET/results/benchmark.json --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/server-info.json s3://$DAI_BUCKET/results/server-info.json --region $DAI_REGION --only-show-errors"
DAI_BENCHMARK_COMMAND=$(send_command "${DAI_INSTANCE_IDS[0]}" "$benchmark_command" 7200)
echo "Running the pre-registered 1,000-input/256-output evaluation ($DAI_BENCHMARK_COMMAND)"
wait_for_command "$DAI_BENCHMARK_COMMAND" "${DAI_INSTANCE_IDS[0]}" 730

collect_rank_artifacts after
download_artifacts

jq -e '.schema == "dai-openai-generation-benchmark.v2" and .prompt_tokens == 1000 and
  .max_tokens == 256 and .repetitions == 10 and .cache_policy == "cold" and
  .cache_flush_count == 11 and ([.runs[] | select(.measured) |
    .output_tokens == 256 and (.output_token_ids | length) == .output_tokens and
    (.output_token_ids | unique | length) > 1] | all)' \
  "$DAI_RESULT_DIR/benchmark.json" >/dev/null
for rank in $(seq 0 $((DAI_WORKERS - 1))); do
  grep -Eq '^NVIDIA L4,' "$DAI_RESULT_DIR/rank-${rank}-gpu-contract.csv"
  DAI_ACTUAL_GPU_MIB=$(awk -F, '{gsub(/ MiB| /, "", $2); print $2}' "$DAI_RESULT_DIR/rank-${rank}-gpu-contract.csv")
  [[ "$DAI_ACTUAL_GPU_MIB" -ge 22000 && "$DAI_ACTUAL_GPU_MIB" -le 24576 ]]
  test -s "$DAI_RESULT_DIR/rank-${rank}-before-network.json"
  test -s "$DAI_RESULT_DIR/rank-${rank}-after-network.json"
done
jq -e --argjson ranks "$DAI_WORKERS" \
  '.tp_size == $ranks and .dp_size == $ranks and .ep_size == $ranks and .enable_dp_attention == true' \
  "$DAI_RESULT_DIR/server-info.json" >/dev/null

cp /Volumes/Extreme/projects/dAi/prototype/results/aws-generation/dai-gen-20260824-065952/baseline.json \
  "$DAI_RESULT_DIR/single-gpu-baseline.json"
python3 "$DAI_ROOT/prototype/analyze_low_vram_ep.py" \
  --manifest "$DAI_RESULT_DIR/manifest.json" \
  --distributed "$DAI_RESULT_DIR/benchmark.json" \
  --baseline "$DAI_RESULT_DIR/single-gpu-baseline.json" \
  --artifact-dir "$DAI_RESULT_DIR" \
  --output "$DAI_RESULT_DIR/report.json"

if [[ -n "$DAI_EXPERT_PLACEMENT_FILE" ]]; then
  echo "Restarting the topology with the optimized expert placement"
  launch_sglang optimized "" /opt/dai/optimized-expert-placement.json
  if ! wait_for_sglang_health optimized; then
    collect_rank_artifacts optimized-startup-failed || true
    download_artifacts || true
    echo "ERROR: optimized expert placement did not initialize." >&2
    exit 1
  fi
  collect_rank_artifacts optimized-before
  optimized_smoke_command="set -euo pipefail; docker exec dai-sglang python3 /opt/dai/generation_benchmark.py --endpoint http://127.0.0.1:30000 --model qwen3-30b-a3b --tokenizer /models/qwen3 --variant low-vram-ep4-optimized-placement-smoke --prompt-file /opt/dai/benchmark-prompt.md --prompt-tokens 1000 --max-tokens 32 --warmups 1 --repetitions 1 --cache-policy cold --nonce dai-generation-v2 --output /opt/dai/results/optimized-smoke.json; aws s3 cp /opt/dai/results/optimized-smoke.json s3://$DAI_BUCKET/results/optimized-smoke.json --region $DAI_REGION --only-show-errors"
  DAI_OPTIMIZED_SMOKE_COMMAND=$(send_command "${DAI_INSTANCE_IDS[0]}" "$optimized_smoke_command" 1800)
  echo "Running the optimized-placement correctness smoke ($DAI_OPTIMIZED_SMOKE_COMMAND)"
  wait_for_command "$DAI_OPTIMIZED_SMOKE_COMMAND" "${DAI_INSTANCE_IDS[0]}" 190
  download_artifacts
  jq -e '[.runs[] | select(.measured) | .output_tokens == 32 and
    (.output_token_ids | length) == .output_tokens and
    (.output_token_ids | unique | length) > 1] | all' \
    "$DAI_RESULT_DIR/optimized-smoke.json" >/dev/null
  optimized_command="set -euo pipefail; docker exec dai-sglang python3 /opt/dai/generation_benchmark.py --endpoint http://127.0.0.1:30000 --model qwen3-30b-a3b --tokenizer /models/qwen3 --variant low-vram-ep4-optimized-placement --prompt-file /opt/dai/benchmark-prompt.md --prompt-tokens 1000 --max-tokens 256 --warmups 2 --repetitions 10 --cache-policy cold --nonce dai-generation-v2 --output /opt/dai/results/optimized-benchmark.json; curl -fsS http://127.0.0.1:30000/get_server_info > /opt/dai/results/optimized-server-info.json; aws s3 cp /opt/dai/results/optimized-benchmark.json s3://$DAI_BUCKET/results/optimized-benchmark.json --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/optimized-server-info.json s3://$DAI_BUCKET/results/optimized-server-info.json --region $DAI_REGION --only-show-errors"
  DAI_OPTIMIZED_COMMAND=$(send_command "${DAI_INSTANCE_IDS[0]}" "$optimized_command" 7200)
  echo "Running the optimized-placement evaluation ($DAI_OPTIMIZED_COMMAND)"
  wait_for_command "$DAI_OPTIMIZED_COMMAND" "${DAI_INSTANCE_IDS[0]}" 730
  collect_rank_artifacts optimized-after
  download_artifacts
  jq -e '[.runs[] | select(.measured) | .output_tokens == 256 and
    (.output_token_ids | length) == .output_tokens and
    (.output_token_ids | unique | length) > 1] | all' \
    "$DAI_RESULT_DIR/optimized-benchmark.json" >/dev/null

  echo "Restarting trivial placement for the post-baseline drift check"
  launch_sglang baseline-post
  if ! wait_for_sglang_health baseline-post; then
    collect_rank_artifacts baseline-post-startup-failed || true
    download_artifacts || true
    echo "ERROR: post-baseline server did not initialize." >&2
    exit 1
  fi
  collect_rank_artifacts baseline-post-before
  baseline_post_command="set -euo pipefail; docker exec dai-sglang python3 /opt/dai/generation_benchmark.py --endpoint http://127.0.0.1:30000 --model qwen3-30b-a3b --tokenizer /models/qwen3 --variant low-vram-ep4-bf16-post --prompt-file /opt/dai/benchmark-prompt.md --prompt-tokens 1000 --max-tokens 256 --warmups 2 --repetitions 10 --cache-policy cold --nonce dai-generation-v2 --output /opt/dai/results/baseline-post-benchmark.json; curl -fsS http://127.0.0.1:30000/get_server_info > /opt/dai/results/baseline-post-server-info.json; aws s3 cp /opt/dai/results/baseline-post-benchmark.json s3://$DAI_BUCKET/results/baseline-post-benchmark.json --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/baseline-post-server-info.json s3://$DAI_BUCKET/results/baseline-post-server-info.json --region $DAI_REGION --only-show-errors"
  DAI_BASELINE_POST_COMMAND=$(send_command "${DAI_INSTANCE_IDS[0]}" "$baseline_post_command" 7200)
  echo "Running the post-baseline evaluation ($DAI_BASELINE_POST_COMMAND)"
  wait_for_command "$DAI_BASELINE_POST_COMMAND" "${DAI_INSTANCE_IDS[0]}" 730
  collect_rank_artifacts baseline-post-after
  download_artifacts

  jq -e '.init_expert_location == "/opt/dai/optimized-expert-placement.json" and
    .ep_dispatch_algorithm == "static" and .enable_eplb == false and
    .ep_num_redundant_experts == 0 and .moe_a2a_backend == "none"' \
    "$DAI_RESULT_DIR/optimized-server-info.json" >/dev/null
  jq -e '.init_expert_location == "trivial" and .enable_eplb == false and
    .ep_num_redundant_experts == 0 and .moe_a2a_backend == "none"' \
    "$DAI_RESULT_DIR/baseline-post-server-info.json" >/dev/null
  python3 "$DAI_ROOT/prototype/compare_expert_placement.py" \
    --trivial-pre "$DAI_RESULT_DIR/benchmark.json" \
    --optimized "$DAI_RESULT_DIR/optimized-benchmark.json" \
    --trivial-post "$DAI_RESULT_DIR/baseline-post-benchmark.json" \
    --optimization-report "$DAI_RESULT_DIR/expert-placement-optimization.json" \
    --output "$DAI_RESULT_DIR/expert-placement-comparison.json"
  jq '{trivial_pre,optimized,trivial_post,baseline_drift_fraction,
    optimized_speedup_vs_bracketed_trivial,predicted_routing}' \
    "$DAI_RESULT_DIR/expert-placement-comparison.json"
fi

if [[ "$DAI_CAPTURE_ROUTING" == "0" ]]; then
  echo "Placement comparison captured at $DAI_RESULT_DIR"
  exit 0
fi

echo "Restarting the same topology for a separate routed-expert trace"
launch_sglang routing-trace "--enable-return-routed-experts"
if ! wait_for_sglang_health routing-trace; then
  collect_routing_artifacts || true
  download_artifacts || true
  echo "ERROR: the instrumented routed-expert server did not initialize." >&2
  exit 1
fi

routing_command="set -euo pipefail; docker exec dai-sglang python3 /opt/dai/generation_benchmark.py --endpoint http://127.0.0.1:30000 --model qwen3-30b-a3b --tokenizer /models/qwen3 --variant low-vram-ep4-routing-trace --prompt-file /opt/dai/benchmark-prompt.md --prompt-tokens 1000 --max-tokens 256 --warmups 2 --repetitions 10 --cache-policy cold --nonce dai-generation-v2 --capture-routed-experts --routed-experts-start-len 0 --routed-expert-layers 48 --routed-expert-top-k 8 --experts-per-layer 128 --expert-parallel-size 4 --expert-routes-output /opt/dai/results/expert-routes.jsonl.gz --expert-summary-output /opt/dai/results/expert-placement-summary.json --output /opt/dai/results/routing-benchmark.json; curl -fsS http://127.0.0.1:30000/get_server_info > /opt/dai/results/routing-server-info.json; aws s3 cp /opt/dai/results/routing-benchmark.json s3://$DAI_BUCKET/results/routing-benchmark.json --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/expert-routes.jsonl.gz s3://$DAI_BUCKET/results/expert-routes.jsonl.gz --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/expert-placement-summary.json s3://$DAI_BUCKET/results/expert-placement-summary.json --region $DAI_REGION --only-show-errors; aws s3 cp /opt/dai/results/routing-server-info.json s3://$DAI_BUCKET/results/routing-server-info.json --region $DAI_REGION --only-show-errors"
DAI_ROUTING_COMMAND=$(send_command "${DAI_INSTANCE_IDS[0]}" "$routing_command" 7200)
echo "Capturing exact request/token/layer expert routes ($DAI_ROUTING_COMMAND)"
if ! wait_for_command "$DAI_ROUTING_COMMAND" "${DAI_INSTANCE_IDS[0]}" 730; then
  collect_routing_artifacts || true
  download_artifacts || true
  echo "ERROR: routed-expert capture failed; retained the instrumented server logs." >&2
  exit 1
fi

collect_routing_artifacts
download_artifacts
gzip -t "$DAI_RESULT_DIR/expert-routes.jsonl.gz"
jq -e '.schema == "dai-openai-generation-benchmark.v2" and
  .routed_expert_capture.enabled == true and
  .routed_expert_capture.num_layers == 48 and
  .routed_expert_capture.top_k == 8 and
  ([.runs[] | select(.measured) | .routed_experts.logical_shape[1:] == [48,8]] | all)' \
  "$DAI_RESULT_DIR/routing-benchmark.json" >/dev/null
jq -e '.schema == "dai-expert-placement-summary.v1" and .request_count == 10 and
  .num_layers == 48 and .top_k == 8 and .experts_per_layer == 128 and
  .expert_parallel_size == 4 and .token_layer_rows > 0 and
  (.layers | length) == 48 and .cross_worker_token_layer_fraction >= 0 and
  .cross_worker_token_layer_fraction <= 1' \
  "$DAI_RESULT_DIR/expert-placement-summary.json" >/dev/null
jq -e '.enable_return_routed_experts == true and .init_expert_location == "trivial" and
  .enable_eplb == false and .ep_num_redundant_experts == 0' \
  "$DAI_RESULT_DIR/routing-server-info.json" >/dev/null

DAI_REPORT_TMP="$DAI_RESULT_DIR/report.json.tmp"
jq --slurpfile routes "$DAI_RESULT_DIR/expert-placement-summary.json" \
  '. + {routing_trace:{artifact:"expert-routes.jsonl.gz",
    summary_artifact:"expert-placement-summary.json",
    request_count:$routes[0].request_count,
    token_layer_rows:$routes[0].token_layer_rows,
    cross_worker_token_layer_fraction:$routes[0].cross_worker_token_layer_fraction,
    worker_fanout_token_layer_counts:$routes[0].worker_fanout_token_layer_counts}}' \
  "$DAI_RESULT_DIR/report.json" > "$DAI_REPORT_TMP"
mv "$DAI_REPORT_TMP" "$DAI_RESULT_DIR/report.json"

jq '{joint_inference_proven,distributed,baseline,speed_ratio,routing_trace}' \
  "$DAI_RESULT_DIR/report.json"
echo "Evaluation captured at $DAI_RESULT_DIR"
