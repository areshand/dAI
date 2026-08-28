data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

data "aws_ssm_parameter" "gpu_ami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
}

data "aws_vpc" "default" {
  default = true
}

locals {
  source_root = "${path.module}/../.."
  common_tags = {
    Project   = "dAI"
    RunId     = var.run_id
    ManagedBy = "OpenTofu"
    ExpiresAt = var.expires_at_utc
    Workload  = "qwen3-low-vram-ep4"
  }
}

resource "aws_s3_bucket" "results" {
  bucket        = "dai-${data.aws_caller_identity.current.account_id}-${var.run_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "results" {
  bucket                  = aws_s3_bucket.results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  rule {
    id     = "expire-run-artifacts"
    status = "Enabled"
    filter {}
    expiration { days = 1 }
  }
}

resource "aws_iam_role" "instance" {
  name = "dai-${var.run_id}-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "artifacts" {
  name = "experiment-artifacts"
  role = aws_iam_role.instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "arn:${data.aws_partition.current.partition}:s3:::${var.model_bucket}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = "arn:${data.aws_partition.current.partition}:s3:::${var.model_bucket}"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject"]
        Resource = "${aws_s3_bucket.results.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.results.arn
      }
    ]
  })
}

resource "aws_iam_instance_profile" "experiment" {
  name = "dai-${var.run_id}"
  role = aws_iam_role.instance.name
}

resource "aws_security_group" "experiment" {
  name_prefix = "dai-${var.run_id}-"
  description = "Run-scoped dAI low-VRAM EP benchmark"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Run-scoped collectives and control plane"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_placement_group" "experiment" {
  name     = "dai-${var.run_id}"
  strategy = "cluster"
}

resource "aws_instance" "worker" {
  count = var.worker_count

  ami                                  = data.aws_ssm_parameter.gpu_ami.value
  instance_type                        = var.instance_type
  subnet_id                            = var.subnet_id
  associate_public_ip_address          = true
  vpc_security_group_ids               = [aws_security_group.experiment.id]
  iam_instance_profile                 = aws_iam_instance_profile.experiment.name
  placement_group                      = aws_placement_group.experiment.name
  instance_initiated_shutdown_behavior = "terminate"
  user_data_base64 = base64gzip(templatefile("${path.module}/cloud-init.tftpl", {
    node_rank                    = count.index
    ttl_minutes                  = var.ttl_minutes
    aws_region                   = var.aws_region
    model_bucket                 = var.model_bucket
    model_prefix                 = var.model_prefix
    result_bucket                = aws_s3_bucket.results.id
    sglang_image                 = var.sglang_image
    generation_benchmark_gz_b64  = base64gzip(file("${local.source_root}/prototype/generation_benchmark.py"))
    qwen3_placement_patch_gz_b64 = base64gzip(file("${local.source_root}/scripts/apply_sglang_qwen3_placement_patch.py"))
    benchmark_prompt_gz_b64 = base64gzip(substr(
      file("${local.source_root}/moe-distributed-experiment-design.md"), 0, 12000
    ))
  }))
  user_data_replace_on_change = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type = "gp3"
    # The pinned SGLang image expands to well over 40 GiB on top of the 57 GiB
    # checkpoint and DLAMI base image. Keep host storage out of the GPU-memory
    # feasibility gate by provisioning enough disposable encrypted scratch.
    volume_size           = 250
    iops                  = 6000
    throughput            = 750
    encrypted             = true
    delete_on_termination = true
    tags                  = local.common_tags
  }

  tags = merge(local.common_tags, {
    Name = "dai-${var.run_id}-ep-${count.index}"
    Role = count.index == 0 ? "ep-coordinator" : "ep-worker"
    Rank = tostring(count.index)
  })

  depends_on = [aws_iam_role_policy_attachment.ssm, aws_iam_role_policy.artifacts]
}
