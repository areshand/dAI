variable "aws_profile" {
  description = "Explicit AWS CLI profile used for the experiment."
  type        = string
  default     = "mi:scratchpad"
}

variable "aws_region" {
  description = "AWS region used by the experiment."
  type        = string
  default     = "us-west-2"
}

variable "run_id" {
  description = "Unique lower-case run identifier used on every resource."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{5,48}$", var.run_id))
    error_message = "run_id must be 6-49 lower-case letters, digits, or hyphens."
  }
}

variable "expires_at_utc" {
  description = "Hard EventBridge termination deadline in RFC3339 UTC form."
  type        = string

  validation {
    condition     = can(formatdate("YYYY-MM-DD'T'hh:mm:ss'Z'", var.expires_at_utc))
    error_message = "expires_at_utc must be an RFC3339 timestamp."
  }
}

variable "ttl_minutes" {
  description = "Independent instance-local self-termination delay."
  type        = number
  default     = 180

  validation {
    condition     = var.ttl_minutes >= 30 && var.ttl_minutes <= 360
    error_message = "ttl_minutes must be between 30 and 360."
  }
}

variable "near_subnet_id" {
  description = "Public subnet for coordinator and near worker."
  type        = string
}

variable "far_subnet_id" {
  description = "Public subnet in a different AZ for the far worker."
  type        = string
}

variable "instance_type" {
  description = "Identical instance type for all first-phase nodes."
  type        = string
  default     = "c7i.xlarge"
}

variable "worker_port" {
  description = "Private-VPC synthetic expert service port."
  type        = number
  default     = 50123
}
