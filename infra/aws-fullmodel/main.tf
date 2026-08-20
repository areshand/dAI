data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

data "aws_subnet" "near" {
  id = var.near_subnet_id
}

data "aws_subnet" "far" {
  id = var.far_subnet_id
}

locals {
  common_tags = {
    Project   = "dAI"
    RunId     = var.run_id
    ManagedBy = "OpenTofu"
    ExpiresAt = var.expires_at_utc
    Workload  = "qwen3-full-model"
  }
  source_root = "${path.module}/../.."
  worker_unit = <<-UNIT
    [Unit]
    Description=dAI real Qwen expert worker
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=simple
    ExecStart=/opt/dai/venv/bin/python /opt/dai/real_expert_probe.py serve --expert /opt/dai/expert.safetensors --bind 0.0.0.0 --port ${var.worker_port} --warmup 3
    Restart=always
    RestartSec=2

    [Install]
    WantedBy=multi-user.target
  UNIT
  node_user_data = {
    coordinator = templatefile("${path.module}/cloud-init.tftpl", {
      role                                = "coordinator"
      ttl_minutes                         = var.ttl_minutes
      aws_region                          = var.aws_region
      model_bucket                        = var.model_bucket
      model_prefix                        = var.model_prefix
      expert_key                          = var.expert_key
      real_expert_probe_gz_b64            = base64gzip(file("${local.source_root}/prototype/real_expert_probe.py"))
      full_model_remote_expert_gz_b64     = base64gzip(file("${local.source_root}/prototype/full_model_remote_expert.py"))
      full_model_multi_worker_eval_gz_b64 = base64gzip(file("${local.source_root}/prototype/full_model_multi_worker_eval.py"))
      worker_unit_b64                     = ""
    })
    near = templatefile("${path.module}/cloud-init.tftpl", {
      role                                = "worker"
      ttl_minutes                         = var.ttl_minutes
      aws_region                          = var.aws_region
      model_bucket                        = var.model_bucket
      model_prefix                        = var.model_prefix
      expert_key                          = var.expert_key
      real_expert_probe_gz_b64            = base64gzip(file("${local.source_root}/prototype/real_expert_probe.py"))
      full_model_remote_expert_gz_b64     = base64gzip(file("${local.source_root}/prototype/full_model_remote_expert.py"))
      full_model_multi_worker_eval_gz_b64 = base64gzip(file("${local.source_root}/prototype/full_model_multi_worker_eval.py"))
      worker_unit_b64                     = base64encode(replace(local.worker_unit, "WORKER_ID", "near"))
    })
    far = templatefile("${path.module}/cloud-init.tftpl", {
      role                                = "worker"
      ttl_minutes                         = var.ttl_minutes
      aws_region                          = var.aws_region
      model_bucket                        = var.model_bucket
      model_prefix                        = var.model_prefix
      expert_key                          = var.expert_key
      real_expert_probe_gz_b64            = base64gzip(file("${local.source_root}/prototype/real_expert_probe.py"))
      full_model_remote_expert_gz_b64     = base64gzip(file("${local.source_root}/prototype/full_model_remote_expert.py"))
      full_model_multi_worker_eval_gz_b64 = base64gzip(file("${local.source_root}/prototype/full_model_multi_worker_eval.py"))
      worker_unit_b64                     = base64encode(replace(local.worker_unit, "WORKER_ID", "far"))
    })
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
  description = "Run-scoped dAI full-model traffic"
  vpc_id      = data.aws_subnet.near.vpc_id

  ingress {
    description = "Real expert RPC within the experiment group"
    from_port   = var.worker_port
    to_port     = var.worker_port
    protocol    = "tcp"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    precondition {
      condition     = data.aws_subnet.near.vpc_id == data.aws_subnet.far.vpc_id
      error_message = "near and far subnets must be in the same VPC."
    }
    precondition {
      condition     = data.aws_subnet.near.availability_zone != data.aws_subnet.far.availability_zone
      error_message = "far subnet must be in a different Availability Zone."
    }
  }
}

resource "aws_placement_group" "near" {
  name     = "dai-${var.run_id}-near"
  strategy = "cluster"
}

resource "aws_instance" "coordinator" {
  ami                                  = data.aws_ssm_parameter.al2023_ami.value
  instance_type                        = var.coordinator_instance_type
  subnet_id                            = var.near_subnet_id
  placement_group                      = aws_placement_group.near.name
  associate_public_ip_address          = true
  vpc_security_group_ids               = [aws_security_group.experiment.id]
  iam_instance_profile                 = aws_iam_instance_profile.experiment.name
  instance_initiated_shutdown_behavior = "terminate"
  user_data_base64                     = base64gzip(local.node_user_data.coordinator)
  user_data_replace_on_change          = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 100
    iops                  = 6000
    throughput            = 500
    encrypted             = true
    delete_on_termination = true
    tags                  = local.common_tags
  }

  tags       = { Name = "dai-${var.run_id}-coordinator", Role = "full-model-coordinator" }
  depends_on = [aws_iam_role_policy_attachment.ssm, aws_iam_role_policy.artifacts]
}

resource "aws_instance" "near" {
  ami                                  = data.aws_ssm_parameter.al2023_ami.value
  instance_type                        = var.worker_instance_type
  subnet_id                            = var.near_subnet_id
  placement_group                      = aws_placement_group.near.name
  associate_public_ip_address          = true
  vpc_security_group_ids               = [aws_security_group.experiment.id]
  iam_instance_profile                 = aws_iam_instance_profile.experiment.name
  instance_initiated_shutdown_behavior = "terminate"
  user_data_base64                     = base64gzip(local.node_user_data.near)
  user_data_replace_on_change          = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 20
    encrypted             = true
    delete_on_termination = true
    tags                  = local.common_tags
  }

  tags       = { Name = "dai-${var.run_id}-near", Role = "real-expert-near" }
  depends_on = [aws_iam_role_policy_attachment.ssm, aws_iam_role_policy.artifacts]
}

resource "aws_instance" "far" {
  ami                                  = data.aws_ssm_parameter.al2023_ami.value
  instance_type                        = var.worker_instance_type
  subnet_id                            = var.far_subnet_id
  associate_public_ip_address          = true
  vpc_security_group_ids               = [aws_security_group.experiment.id]
  iam_instance_profile                 = aws_iam_instance_profile.experiment.name
  instance_initiated_shutdown_behavior = "terminate"
  user_data_base64                     = base64gzip(local.node_user_data.far)
  user_data_replace_on_change          = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 20
    encrypted             = true
    delete_on_termination = true
    tags                  = local.common_tags
  }

  tags       = { Name = "dai-${var.run_id}-far", Role = "real-expert-far" }
  depends_on = [aws_iam_role_policy_attachment.ssm, aws_iam_role_policy.artifacts]
}
