#!/usr/bin/env bash
set -euo pipefail

DAI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DAI_PROFILE=${DAI_AWS_PROFILE:-mi:scratchpad}
DAI_REGION=${DAI_AWS_REGION:-us-west-2}

exec python3 "$DAI_ROOT/prototype/aws_cost_estimate.py" \
  --profile "$DAI_PROFILE" \
  --region "$DAI_REGION"
