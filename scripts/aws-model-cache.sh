#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--upload" ) ]]; then
  echo "usage: $0 [--upload]" >&2
  exit 2
fi

DAI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DAI_PROFILE=${DAI_AWS_PROFILE:-mi:scratchpad}
DAI_REGION=${DAI_AWS_REGION:-us-west-2}
DAI_MODEL_LOCAL=${DAI_MODEL_LOCAL:-$DAI_ROOT/model-cache/qwen3-30b-a3b}
DAI_ACCOUNT=$(aws sts get-caller-identity --profile "$DAI_PROFILE" --query Account --output text)
DAI_BUCKET=${DAI_MODEL_BUCKET:-dai-${DAI_ACCOUNT}-model-cache-${DAI_REGION}}

if ! aws s3api head-bucket --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --bucket "$DAI_BUCKET" 2>/dev/null; then
  aws s3api create-bucket --profile "$DAI_PROFILE" --region "$DAI_REGION" \
    --bucket "$DAI_BUCKET" --create-bucket-configuration "LocationConstraint=$DAI_REGION"
fi
aws s3api put-public-access-block --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --bucket "$DAI_BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --bucket "$DAI_BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
aws s3api put-bucket-tagging --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --bucket "$DAI_BUCKET" --tagging \
  'TagSet=[{Key=Project,Value=dAI},{Key=Purpose,Value=model-cache},{Key=ManagedBy,Value=Codex}]'
aws s3api put-bucket-lifecycle-configuration --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --bucket "$DAI_BUCKET" --lifecycle-configuration \
  '{"Rules":[{"ID":"expire-model-cache","Status":"Enabled","Filter":{"Prefix":""},"Expiration":{"Days":7},"AbortIncompleteMultipartUpload":{"DaysAfterInitiation":1}}]}'

if [[ ${1:-} == "--upload" ]]; then
  if [[ ! -s "$DAI_MODEL_LOCAL/model.safetensors.index.json" ]]; then
    echo "ERROR: model checkpoint is missing at $DAI_MODEL_LOCAL." >&2
    exit 1
  fi
  aws s3 sync "$DAI_MODEL_LOCAL" "s3://$DAI_BUCKET/qwen3-30b-a3b/" \
    --profile "$DAI_PROFILE" --region "$DAI_REGION" --exclude '.cache/*' --only-show-errors
  aws s3 cp "$DAI_ROOT/prototype/artifacts/qwen3-layer0-expert53.safetensors" \
    "s3://$DAI_BUCKET/artifacts/qwen3-layer0-expert53.safetensors" \
    --profile "$DAI_PROFILE" --region "$DAI_REGION" --only-show-errors
fi

# JMESPath backticks are literals and must not expand in the local shell.
# shellcheck disable=SC2016
aws s3api list-objects-v2 --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --bucket "$DAI_BUCKET" \
  --query '{Objects:length(not_null(Contents,`[]`)),Bytes:sum(not_null(Contents[].Size,`[]`))}' \
  --output json
echo "Protected seven-day model cache: $DAI_BUCKET"
