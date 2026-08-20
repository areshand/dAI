#!/usr/bin/env bash
set -euo pipefail

DAI_PROFILE=${DAI_AWS_PROFILE:-mi:scratchpad}
DAI_REGION=${DAI_AWS_REGION:-us-west-2}

# JMESPath uses backticks as literals; this is intentionally single-quoted.
# shellcheck disable=SC2016
aws ec2 describe-instances --profile "$DAI_PROFILE" --region "$DAI_REGION" \
  --filters Name=tag:Project,Values=dAI Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
  --query 'Reservations[].Instances[].{RunId:Tags[?Key==`RunId`]|[0].Value,Id:InstanceId,State:State.Name,Type:InstanceType,AZ:Placement.AvailabilityZone,Name:Tags[?Key==`Name`]|[0].Value,ExpiresAt:Tags[?Key==`ExpiresAt`]|[0].Value}' \
  --output table
