#!/usr/bin/env bash
set -euo pipefail

DAI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DAI_TOFU_DIR="$DAI_ROOT/infra/aws-generation"
DAI_PROFILE=${DAI_AWS_PROFILE:-mi:scratchpad}
DAI_REGION=${DAI_AWS_REGION:-us-west-2}
DAI_INSTANCE_TYPE=${DAI_AWS_GPU_INSTANCE_TYPE:-g5.12xlarge}
DAI_PREFERRED_AZ=${DAI_AWS_AVAILABILITY_ZONE:-}
DAI_TP_SIZE=${DAI_TP_SIZE:-4}
DAI_MEM_FRACTION_STATIC=${DAI_MEM_FRACTION_STATIC:-0.82}
DAI_SGLANG_IMAGE=${DAI_SGLANG_IMAGE:-lmsysorg/sglang:v0.5.16}
DAI_MODEL_PREFIX=${DAI_MODEL_PREFIX:-qwen3-30b-a3b}
DAI_MODEL_LOCAL=${DAI_MODEL_LOCAL:-$DAI_ROOT/model-cache/qwen3-30b-a3b}
DAI_MODEL_BUCKET=${DAI_MODEL_BUCKET:-}
DAI_QUALITY_DATASET_LOCAL=${DAI_QUALITY_DATASET_LOCAL:-$DAI_ROOT/prototype/quality/quality-smoke-v1.jsonl}
DAI_QUALITY_REPETITIONS=${DAI_QUALITY_REPETITIONS:-1}
DAI_QUALITY_MARGIN=${DAI_QUALITY_MARGIN:-0.02}
DAI_QUALITY_MIN_CASES=${DAI_QUALITY_MIN_CASES:-100}
DAI_TARGET_MEAN_TPS=${DAI_TARGET_MEAN_TPS:-100}
DAI_MAX_EVENT_GAP_MS=${DAI_MAX_EVENT_GAP_MS:-100}
DAI_MAX_TTFT_MS=${DAI_MAX_TTFT_MS:-250}
DAI_TTL_MINUTES=${DAI_AWS_TTL_MINUTES:-180}
DAI_MAX_HOURLY_USD=${DAI_MAX_HOURLY_USD:-8}
DAI_EXPECTED_HOURLY_USD=${DAI_EXPECTED_HOURLY_USD:-}
DAI_VARIANT_SET=${DAI_VARIANT_SET:-qualification}
# SGLang documents Triton as a deterministic backend for Qwen3-30B-A3B. The
# FlashInfer deterministic path overflowed its prefill workspace on G7e
# (Blackwell), so qualification runs pin Triton instead of silently falling
# back to a non-reproducible backend.
DAI_COMMON_SERVER_ARGS=${DAI_COMMON_SERVER_ARGS:---random-seed 1234 --enable-deterministic-inference --attention-backend triton}
if [[ "$DAI_VARIANT_SET" != "qualification" && "$DAI_VARIANT_SET" != "all" && "$DAI_VARIANT_SET" != "quality" && "$DAI_VARIANT_SET" != "draft-profile" ]]; then
  echo "ERROR: DAI_VARIANT_SET must be qualification, all, quality, or draft-profile." >&2
  exit 1
fi
DAI_RUN_ID=${DAI_RUN_ID:-dai-gen-$(date -u +%Y%m%d-%H%M%S)}
DAI_RESULT_DIR="$DAI_ROOT/prototype/results/aws-generation/$DAI_RUN_ID"
DAI_TFVARS="$DAI_TOFU_DIR/run.auto.tfvars.json"
DAI_PLAN=$(mktemp /private/tmp/dai-generation-plan.XXXXXX)
DAI_COMMAND_INPUT=$(mktemp /private/tmp/dai-generation-command.XXXXXX.json)
DAI_CLEANUP_FAILED=0
DAI_RESULT_FILES=()
DAI_QUALITY_RESULT_FILES=()

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

  rm -f "$DAI_PLAN" "$DAI_COMMAND_INPUT" "$DAI_TFVARS"
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

send_command() {
  local command=$1
  local timeout_seconds=$2
  local encoded_command
  local bash_command
  # AWS-RunShellScript uses /bin/sh. On Ubuntu that is dash, which rejects
  # `set -o pipefail`; encode the payload and explicitly execute it with Bash.
  encoded_command=$(printf '%s' "$command" | base64 | tr -d '\n')
  bash_command="echo '$encoded_command' | base64 -d | bash"
  jq -n --arg instance "$DAI_INSTANCE" --arg command "$bash_command" --arg bucket "$DAI_BUCKET" \
    --argjson timeout "$timeout_seconds" \
    '{InstanceIds:[$instance],DocumentName:"AWS-RunShellScript",TimeoutSeconds:$timeout,
      Parameters:{commands:[$command]},OutputS3BucketName:$bucket,OutputS3KeyPrefix:"ssm-output"}' \
    > "$DAI_COMMAND_INPUT"
  aws ssm send-command --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --cli-input-json "file://$DAI_COMMAND_INPUT" --query Command.CommandId --output text
}

run_variant() {
  local variant=$1
  local extra_args=$2
  local remote_result="/opt/dai/results/${variant}.json"
  local start_command
  local benchmark_command
  local spec_probe_command=""
  local quality_command=""

  start_command="set -euo pipefail; docker rm -f dai-sglang >/dev/null 2>&1 || true; docker run -d --name dai-sglang --gpus all --ipc=host --network host --shm-size 32g -v /opt/dai/model:/models/qwen3:ro -v /opt/dai:/opt/dai -v /opt/dai/hf-cache:/root/.cache/huggingface '$DAI_SGLANG_IMAGE' python3 -m sglang.launch_server --model-path /models/qwen3 --served-model-name qwen3-30b-a3b --host 127.0.0.1 --port 30000 --tp-size '$DAI_TP_SIZE' --dtype bfloat16 --mem-fraction-static '$DAI_MEM_FRACTION_STATIC' $DAI_COMMON_SERVER_ARGS $extra_args; for i in \$(seq 1 240); do if curl -fsS http://127.0.0.1:30000/health >/dev/null; then exit 0; fi; sleep 5; done; docker logs dai-sglang; exit 1"
  DAI_COMMAND_ID=$(send_command "$start_command" 1800)
  echo "Starting $variant server ($DAI_COMMAND_ID)"
  wait_for_command "$DAI_COMMAND_ID" "$DAI_INSTANCE" 190

  if [[ "$variant" == eagle3 || "$variant" == eagle3-compile || "$variant" == standalone ]]; then
    spec_probe_command="docker exec dai-sglang python3 /opt/dai/native_spec_probe.py --endpoint http://127.0.0.1:30000 --tokenizer /models/qwen3 --variant '$variant' --prompt-file /opt/dai/benchmark-prompt.md --prompt-tokens 1000 --max-tokens 256 --warmups 1 --repetitions 5 --nonce dai-spec-probe-v1 --output /opt/dai/results/${variant}-spec-probe.json; aws s3 cp /opt/dai/results/${variant}-spec-probe.json 's3://$DAI_BUCKET/results/${variant}-spec-probe.json' --region '$DAI_REGION' --only-show-errors;"
  fi
  if [[ "$DAI_VARIANT_SET" == "quality" ]]; then
    quality_command="docker exec dai-sglang python3 /opt/dai/quality_benchmark.py --endpoint http://127.0.0.1:30000 --model qwen3-30b-a3b --variant '$variant' --dataset /opt/dai/quality/suite.jsonl --repetitions '$DAI_QUALITY_REPETITIONS' --output /opt/dai/results/${variant}-quality.json; aws s3 cp /opt/dai/results/${variant}-quality.json 's3://$DAI_BUCKET/results/${variant}-quality.json' --region '$DAI_REGION' --only-show-errors;"
  fi
  benchmark_command="set -euo pipefail; docker exec dai-sglang python3 /opt/dai/generation_benchmark.py --endpoint http://127.0.0.1:30000 --model qwen3-30b-a3b --tokenizer /models/qwen3 --variant '$variant' --prompt-file /opt/dai/benchmark-prompt.md --prompt-tokens 1000 --max-tokens 256 --warmups 2 --repetitions 10 --nonce dai-generation-v2 --output '$remote_result'; $quality_command $spec_probe_command curl -fsS http://127.0.0.1:30000/get_server_info > /opt/dai/results/${variant}-server-info.json; docker logs dai-sglang > /opt/dai/results/${variant}-server.log 2>&1; aws s3 cp '$remote_result' 's3://$DAI_BUCKET/results/${variant}.json' --region '$DAI_REGION' --only-show-errors; aws s3 cp /opt/dai/results/${variant}-server-info.json 's3://$DAI_BUCKET/results/${variant}-server-info.json' --region '$DAI_REGION' --only-show-errors; aws s3 cp /opt/dai/results/${variant}-server.log 's3://$DAI_BUCKET/results/${variant}-server.log' --region '$DAI_REGION' --only-show-errors; docker rm -f dai-sglang >/dev/null"
  DAI_COMMAND_ID=$(send_command "$benchmark_command" 7200)
  echo "Benchmarking $variant ($DAI_COMMAND_ID)"
  wait_for_command "$DAI_COMMAND_ID" "$DAI_INSTANCE" 720
  aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    "s3://$DAI_BUCKET/results/${variant}.json" "$DAI_RESULT_DIR/${variant}.json" --only-show-errors
  aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    "s3://$DAI_BUCKET/results/${variant}-server-info.json" "$DAI_RESULT_DIR/${variant}-server-info.json" --only-show-errors
  aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    "s3://$DAI_BUCKET/results/${variant}-server.log" "$DAI_RESULT_DIR/${variant}-server.log" --only-show-errors
  if [[ -n "$spec_probe_command" ]]; then
    aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
      "s3://$DAI_BUCKET/results/${variant}-spec-probe.json" "$DAI_RESULT_DIR/${variant}-spec-probe.json" --only-show-errors
  fi
  jq -e '.schema == "dai-openai-generation-benchmark.v2" and .prompt_tokens == 1000 and .max_tokens == 256 and .repetitions == 10' \
    "$DAI_RESULT_DIR/${variant}.json" >/dev/null
  DAI_RESULT_FILES+=("$DAI_RESULT_DIR/${variant}.json")
  if [[ "$DAI_VARIANT_SET" == "quality" ]]; then
    aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
      "s3://$DAI_BUCKET/results/${variant}-quality.json" "$DAI_RESULT_DIR/${variant}-quality.json" --only-show-errors
    jq -e '.schema == "dai-quality-benchmark.v1" and .summary.cases > 0' \
      "$DAI_RESULT_DIR/${variant}-quality.json" >/dev/null
    DAI_QUALITY_RESULT_FILES+=("$DAI_RESULT_DIR/${variant}-quality.json")
    jq '{variant,summary}' "$DAI_RESULT_DIR/${variant}-quality.json"
  fi
  jq '{variant,summary}' "$DAI_RESULT_DIR/${variant}.json"
}

run_draft_profile() {
  local variant="draft-gpu"
  local remote_result="/opt/dai/results/${variant}.json"
  local server_variant="draft-sglang-gpu"
  local remote_server_result="/opt/dai/results/${server_variant}.json"
  local profile_command

  # Deterministic inference selects a persistent Triton matmul whose shared
  # memory requirement exceeds an L4's per-block limit. This cell measures
  # draft-side latency only; correctness remains target-authoritative in the
  # end-to-end speculative runs, so use ordinary seeded kernels here.
  profile_command="set -euo pipefail; docker rm -f dai-sglang dai-draft-profile >/dev/null 2>&1 || true; docker run --rm --name dai-draft-profile --gpus all --ipc=host --network host --shm-size 8g -v /opt/dai:/opt/dai -v /opt/dai/hf-cache:/root/.cache/huggingface '$DAI_SGLANG_IMAGE' python3 /opt/dai/draft_profile.py --model Qwen/Qwen3-0.6B --device cuda --dtype float16 --prompt-tokens 256 --warmups 2 --repetitions 10 --round-sizes 2,4,8,16 --output '$remote_result'; aws s3 cp '$remote_result' 's3://$DAI_BUCKET/results/${variant}.json' --region '$DAI_REGION' --only-show-errors; docker run -d --name dai-sglang --gpus all --ipc=host --network host --shm-size 8g -v /opt/dai:/opt/dai -v /opt/dai/hf-cache:/root/.cache/huggingface '$DAI_SGLANG_IMAGE' python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --served-model-name qwen3-0.6b --host 127.0.0.1 --port 30000 --tp-size 1 --dtype float16 --mem-fraction-static 0.70 --cuda-graph-backend-prefill disabled --random-seed 1234 --attention-backend triton; for i in \$(seq 1 240); do if curl -fsS http://127.0.0.1:30000/health >/dev/null; then break; fi; if ! docker inspect -f '{{.State.Running}}' dai-sglang 2>/dev/null | grep -q true; then docker logs dai-sglang; exit 1; fi; if [[ \$i -eq 240 ]]; then docker logs dai-sglang; exit 1; fi; sleep 5; done; docker exec dai-sglang python3 /opt/dai/generation_benchmark.py --endpoint http://127.0.0.1:30000 --model qwen3-0.6b --tokenizer Qwen/Qwen3-0.6B --variant '$server_variant' --prompt-file /opt/dai/benchmark-prompt.md --prompt-tokens 1000 --max-tokens 256 --warmups 2 --repetitions 10 --nonce dai-draft-sglang-v1 --output '$remote_server_result'; curl -fsS http://127.0.0.1:30000/get_server_info > /opt/dai/results/${server_variant}-server-info.json; docker logs dai-sglang > /opt/dai/results/${server_variant}-server.log 2>&1; aws s3 cp '$remote_server_result' 's3://$DAI_BUCKET/results/${server_variant}.json' --region '$DAI_REGION' --only-show-errors; aws s3 cp /opt/dai/results/${server_variant}-server-info.json 's3://$DAI_BUCKET/results/${server_variant}-server-info.json' --region '$DAI_REGION' --only-show-errors; aws s3 cp /opt/dai/results/${server_variant}-server.log 's3://$DAI_BUCKET/results/${server_variant}-server.log' --region '$DAI_REGION' --only-show-errors; docker rm -f dai-sglang >/dev/null"
  DAI_COMMAND_ID=$(send_command "$profile_command" 3600)
  echo "Profiling Qwen3-0.6B draft GPU ($DAI_COMMAND_ID)"
  wait_for_command "$DAI_COMMAND_ID" "$DAI_INSTANCE" 360
  aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    "s3://$DAI_BUCKET/results/${variant}.json" "$DAI_RESULT_DIR/${variant}.json" --only-show-errors
  aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    "s3://$DAI_BUCKET/results/${server_variant}.json" "$DAI_RESULT_DIR/${server_variant}.json" --only-show-errors
  aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    "s3://$DAI_BUCKET/results/${server_variant}-server-info.json" "$DAI_RESULT_DIR/${server_variant}-server-info.json" --only-show-errors
  aws s3 cp --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    "s3://$DAI_BUCKET/results/${server_variant}-server.log" "$DAI_RESULT_DIR/${server_variant}-server.log" --only-show-errors
  jq '{host,platform,device,dtype,model,summary}' "$DAI_RESULT_DIR/${variant}.json"
  jq '{variant,summary}' "$DAI_RESULT_DIR/${server_variant}.json"
}

if tofu -chdir="$DAI_TOFU_DIR" state list 2>/dev/null | grep -q .; then
  echo "ERROR: infra/aws-generation already has managed resources." >&2
  exit 1
fi
DAI_ACCOUNT=$(aws sts get-caller-identity --profile "$DAI_PROFILE" --query Account --output text)
DAI_MODEL_BUCKET=${DAI_MODEL_BUCKET:-dai-${DAI_ACCOUNT}-model-cache-${DAI_REGION}}

if [[ "$DAI_VARIANT_SET" != "draft-profile" ]]; then
  if [[ ! -s "$DAI_MODEL_LOCAL/model.safetensors.index.json" ]]; then
    echo "ERROR: local pinned model is missing at $DAI_MODEL_LOCAL." >&2
    exit 1
  fi
  echo "Verifying that the protected S3 model cache exactly covers the local checkpoint"
  DAI_SYNC_DIFF=$(aws s3 sync "$DAI_MODEL_LOCAL" "s3://$DAI_MODEL_BUCKET/$DAI_MODEL_PREFIX/" \
    --profile "$DAI_PROFILE" --region "$DAI_REGION" --exclude '.cache/*' --delete --dryrun)
  if [[ -n "$DAI_SYNC_DIFF" ]]; then
    echo "ERROR: S3 model cache is incomplete or differs from the local checkpoint." >&2
    echo "$DAI_SYNC_DIFF" >&2
    exit 1
  fi
fi
if [[ "$DAI_VARIANT_SET" == "quality" && ! -s "$DAI_QUALITY_DATASET_LOCAL" ]]; then
  echo "ERROR: quality dataset is missing at $DAI_QUALITY_DATASET_LOCAL." >&2
  exit 1
fi

if [[ -n "$DAI_EXPECTED_HOURLY_USD" ]]; then
  DAI_HOURLY_USD=$DAI_EXPECTED_HOURLY_USD
elif [[ "$DAI_INSTANCE_TYPE" == "g5.12xlarge" ]]; then
  # Published AWS on-demand rate for Linux in us-west-2. The account role does
  # not grant pricing:GetProducts, so non-default types require an explicit rate.
  DAI_HOURLY_USD=5.672
else
  echo "ERROR: set DAI_EXPECTED_HOURLY_USD for non-default type $DAI_INSTANCE_TYPE." >&2
  exit 1
fi
if ! awk -v price="$DAI_HOURLY_USD" -v ceiling="$DAI_MAX_HOURLY_USD" 'BEGIN { exit !(price <= ceiling) }'; then
  echo "ERROR: $DAI_INSTANCE_TYPE costs \$$DAI_HOURLY_USD/hour, above DAI_MAX_HOURLY_USD=\$$DAI_MAX_HOURLY_USD." >&2
  exit 1
fi

DAI_SUBNET=""
if [[ -n "$DAI_PREFERRED_AZ" ]]; then
  DAI_SUBNET=$(aws ec2 describe-subnets --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --filters Name=default-for-az,Values=true "Name=availability-zone,Values=$DAI_PREFERRED_AZ" \
    --query 'Subnets[0].SubnetId' --output text)
  if [[ "$DAI_SUBNET" == "None" || -z "$DAI_SUBNET" ]]; then
    echo "ERROR: no default subnet exists in requested AZ $DAI_PREFERRED_AZ." >&2
    exit 1
  fi
fi

DAI_EXPIRES_AT=$(python3 -c 'import datetime,sys; print((datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=int(sys.argv[1]))).replace(microsecond=0).isoformat().replace("+00:00","Z"))' "$DAI_TTL_MINUTES")
mkdir -p "$DAI_RESULT_DIR"
jq -n --arg profile "$DAI_PROFILE" --arg region "$DAI_REGION" --arg run_id "$DAI_RUN_ID" \
  --arg expires "$DAI_EXPIRES_AT" --arg subnet "$DAI_SUBNET" --arg model_bucket "$DAI_MODEL_BUCKET" \
  --arg model_prefix "$DAI_MODEL_PREFIX" --arg instance_type "$DAI_INSTANCE_TYPE" \
  --arg sglang_image "$DAI_SGLANG_IMAGE" --arg quality_dataset_path "$DAI_QUALITY_DATASET_LOCAL" \
  --argjson ttl "$DAI_TTL_MINUTES" \
  --argjson sync_full_model "$([[ "$DAI_VARIANT_SET" == "draft-profile" ]] && echo false || echo true)" \
  '{aws_profile:$profile,aws_region:$region,run_id:$run_id,expires_at_utc:$expires,
    ttl_minutes:$ttl,subnet_id:$subnet,model_bucket:$model_bucket,model_prefix:$model_prefix,
    instance_type:$instance_type,sglang_image:$sglang_image,sync_full_model:$sync_full_model,
    quality_dataset_path:$quality_dataset_path}' > "$DAI_TFVARS"

echo "Planning $DAI_RUN_ID: $DAI_INSTANCE_TYPE at \$$DAI_HOURLY_USD/hour, hard expiry $DAI_EXPIRES_AT"
tofu -chdir="$DAI_TOFU_DIR" init -input=false
tofu -chdir="$DAI_TOFU_DIR" plan -input=false -out="$DAI_PLAN"
tofu -chdir="$DAI_TOFU_DIR" apply -input=false "$DAI_PLAN"

DAI_INSTANCE=$(tofu -chdir="$DAI_TOFU_DIR" output -raw instance_id)
DAI_BUCKET=$(tofu -chdir="$DAI_TOFU_DIR" output -raw result_bucket)

echo "Waiting for the GPU node to become SSM-online"
for _ in $(seq 1 72); do
  DAI_PING=$(aws ssm describe-instance-information --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --filters "Key=InstanceIds,Values=$DAI_INSTANCE" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [[ "$DAI_PING" == "Online" ]] && break
  sleep 5
done
if [[ "${DAI_PING:-}" != "Online" ]]; then
  echo "ERROR: GPU node did not become SSM-online." >&2
  exit 1
fi

# The command substitution is intentionally passed literally for execution on EC2.
# shellcheck disable=SC2016
DAI_READY_COMMAND=$(send_command 'for i in $(seq 1 300); do test -f /opt/dai/ready && systemctl is-active --quiet dai-ttl-terminate.timer && exit 0; sleep 5; done; exit 1' 1800)
echo "Waiting for S3 model sync, pinned SGLang image, and active TTL timer"
wait_for_command "$DAI_READY_COMMAND" "$DAI_INSTANCE" 190

if [[ "$DAI_VARIANT_SET" == "draft-profile" ]]; then
  run_draft_profile
  echo "Draft profiles captured: $DAI_RESULT_DIR/draft-gpu.json and $DAI_RESULT_DIR/draft-sglang-gpu.json"
  exit 0
fi

case "$DAI_VARIANT_SET" in
  qualification)
    run_variant baseline ""
    run_variant ngram "--speculative-algorithm NGRAM"
    run_variant eagle3-compile "--context-length 2048 --enable-fused-qk-norm-rope --enable-torch-compile --torch-compile-max-bs 1 --speculative-algorithm EAGLE3 --speculative-draft-model-path yschoi31/Qwen3-30B-A3B-Eagle3 --speculative-draft-model-revision 74ef9c5 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
    run_variant standalone "--context-length 2048 --speculative-algorithm STANDALONE --speculative-draft-model-path Qwen/Qwen3-0.6B --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5"
    ;;
  all)
    run_variant baseline ""
    run_variant ngram "--speculative-algorithm NGRAM"
    run_variant ngram16 "--speculative-algorithm NGRAM --speculative-num-draft-tokens 16 --speculative-ngram-max-bfs-breadth 10"
    run_variant ngram32 "--speculative-algorithm NGRAM --speculative-num-draft-tokens 32 --speculative-ngram-max-bfs-breadth 10"
    run_variant eagle3 "--context-length 2048 --speculative-algorithm EAGLE3 --speculative-draft-model-path yschoi31/Qwen3-30B-A3B-Eagle3 --speculative-draft-model-revision 74ef9c5 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
    run_variant eagle3-compile "--context-length 2048 --enable-fused-qk-norm-rope --enable-torch-compile --torch-compile-max-bs 1 --speculative-algorithm EAGLE3 --speculative-draft-model-path yschoi31/Qwen3-30B-A3B-Eagle3 --speculative-draft-model-revision 74ef9c5 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
    run_variant standalone "--context-length 2048 --speculative-algorithm STANDALONE --speculative-draft-model-path Qwen/Qwen3-0.6B --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5"
    ;;
  quality)
    run_variant baseline ""
    run_variant ngram16 "--speculative-algorithm NGRAM --speculative-num-draft-tokens 16 --speculative-ngram-max-bfs-breadth 10"
    run_variant eagle3-compile "--context-length 2048 --enable-fused-qk-norm-rope --enable-torch-compile --torch-compile-max-bs 1 --speculative-algorithm EAGLE3 --speculative-draft-model-path yschoi31/Qwen3-30B-A3B-Eagle3 --speculative-draft-model-revision 74ef9c5 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
    ;;
esac

python3 "$DAI_ROOT/prototype/compare_generation_results.py" \
  --output "$DAI_RESULT_DIR/comparison.json" \
  "${DAI_RESULT_FILES[@]}"

echo "Comparison captured: $DAI_RESULT_DIR/comparison.json"
jq . "$DAI_RESULT_DIR/comparison.json"

if [[ "$DAI_VARIANT_SET" == "quality" ]]; then
  quality_args=()
  generation_args=()
  for result in "${DAI_QUALITY_RESULT_FILES[@]}"; do
    quality_args+=(--quality-report "$result")
  done
  for result in "${DAI_RESULT_FILES[@]}"; do
    generation_args+=(--generation-report "$result")
  done
  python3 "$DAI_ROOT/prototype/compare_quality_results.py" \
    "${quality_args[@]}" "${generation_args[@]}" \
    --margin "$DAI_QUALITY_MARGIN" --min-cases "$DAI_QUALITY_MIN_CASES" \
    --target-tps "$DAI_TARGET_MEAN_TPS" --max-event-gap-ms "$DAI_MAX_EVENT_GAP_MS" \
    --max-ttft-ms "$DAI_MAX_TTFT_MS" --output "$DAI_RESULT_DIR/quality-gates.json"
  echo "Quality/speed gates captured: $DAI_RESULT_DIR/quality-gates.json"
fi
