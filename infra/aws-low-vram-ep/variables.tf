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
}

variable "ttl_minutes" {
  type    = number
  default = 180
  validation {
    condition     = var.ttl_minutes >= 60 && var.ttl_minutes <= 240
    error_message = "ttl_minutes must be between 60 and 240."
  }
}

variable "worker_count" {
  type    = number
  default = 4
  validation {
    condition     = var.worker_count == 4
    error_message = "The BF16 Qwen3-30B-A3B experiment is pre-registered for exactly four ranks."
  }
}

variable "instance_type" {
  type    = string
  default = "gr6.4xlarge"
  validation {
    condition     = contains(["g6.4xlarge", "gr6.4xlarge"], var.instance_type)
    error_message = "Only full-L4 AWS shapes below the 24 GiB VRAM ceiling are allowed."
  }
}

variable "subnet_id" {
  type = string
}

variable "model_bucket" {
  type = string
}

variable "model_prefix" {
  type    = string
  default = "qwen3-30b-a3b"
}

variable "sglang_image" {
  type    = string
  default = "lmsysorg/sglang:v0.5.16"
}
