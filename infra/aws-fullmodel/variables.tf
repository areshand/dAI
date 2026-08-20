variable "aws_profile" {
  type    = string
  default = "mi:scratchpad"
}

variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "run_id" {
  type = string
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{5,48}$", var.run_id))
    error_message = "run_id must be 6-49 lower-case letters, digits, or hyphens."
  }
}

variable "expires_at_utc" {
  type = string
  validation {
    condition     = can(formatdate("YYYY-MM-DD'T'hh:mm:ss'Z'", var.expires_at_utc))
    error_message = "expires_at_utc must be an RFC3339 timestamp."
  }
}

variable "ttl_minutes" {
  type    = number
  default = 240
  validation {
    condition     = var.ttl_minutes >= 60 && var.ttl_minutes <= 360
    error_message = "ttl_minutes must be between 60 and 360."
  }
}

variable "near_subnet_id" {
  type = string
}

variable "far_subnet_id" {
  type = string
}

variable "model_bucket" {
  type = string
}

variable "model_prefix" {
  type    = string
  default = "qwen3-30b-a3b"
}

variable "expert_key" {
  type    = string
  default = "artifacts/qwen3-layer0-expert53.safetensors"
}

variable "coordinator_instance_type" {
  type    = string
  default = "r7i.4xlarge"
}

variable "worker_instance_type" {
  type    = string
  default = "c7i.xlarge"
}

variable "worker_port" {
  type    = number
  default = 50126
}
