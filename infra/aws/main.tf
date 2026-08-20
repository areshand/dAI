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
  }

  source_root = "${path.module}/../.."
  worker_unit = <<-UNIT
    [Unit]
    Description=dAI synthetic expert worker
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=simple
    ExecStart=/usr/bin/python3 /opt/dai/two_node_moe.py serve --bind 0.0.0.0 --port ${var.worker_port} --worker-id WORKER_ID
    Restart=always
    RestartSec=2

    [Install]
    WantedBy=multi-user.target
  UNIT
  node_user_data = {
    coordinator = templatefile("${path.module}/cloud-init.tftpl", {
      role                  = "coordinator"
      ttl_minutes           = var.ttl_minutes
      two_node_moe_gz_b64   = base64gzip(file("${local.source_root}/prototype/two_node_moe.py"))
      multi_node_moe_gz_b64 = base64gzip(file("${local.source_root}/prototype/multi_node_moe.py"))
      worker_unit_b64       = ""
    })
    near = templatefile("${path.module}/cloud-init.tftpl", {
      role                  = "worker"
      ttl_minutes           = var.ttl_minutes
      two_node_moe_gz_b64   = base64gzip(file("${local.source_root}/prototype/two_node_moe.py"))
      multi_node_moe_gz_b64 = base64gzip(file("${local.source_root}/prototype/multi_node_moe.py"))
      worker_unit_b64       = base64encode(replace(local.worker_unit, "WORKER_ID", "near"))
    })
    far = templatefile("${path.module}/cloud-init.tftpl", {
      role                  = "worker"
      ttl_minutes           = var.ttl_minutes
      two_node_moe_gz_b64   = base64gzip(file("${local.source_root}/prototype/two_node_moe.py"))
      multi_node_moe_gz_b64 = base64gzip(file("${local.source_root}/prototype/multi_node_moe.py"))
      worker_unit_b64       = base64encode(replace(local.worker_unit, "WORKER_ID", "far"))
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
    expiration {
      days = 1
    }
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

resource "aws_iam_role_policy" "results" {
  name = "result-bucket"
  role = aws_iam_role.instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
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
  description = "Run-scoped dAI experiment traffic"
  vpc_id      = data.aws_subnet.near.vpc_id

  ingress {
    description = "Synthetic expert RPC within the experiment group"
    from_port   = var.worker_port
    to_port     = var.worker_port
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "Directed network probes within the experiment group"
    from_port   = -1
    to_port     = -1
    protocol    = "icmp"
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
  instance_type                        = var.instance_type
  subnet_id                            = var.near_subnet_id
  placement_group                      = aws_placement_group.near.name
  associate_public_ip_address          = true
  vpc_security_group_ids               = [aws_security_group.experiment.id]
  iam_instance_profile                 = aws_iam_instance_profile.experiment.name
  instance_initiated_shutdown_behavior = "terminate"
  user_data                            = local.node_user_data.coordinator
  user_data_replace_on_change          = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 16
    encrypted             = true
    delete_on_termination = true
    tags                  = local.common_tags
  }

  tags       = { Name = "dai-${var.run_id}-coordinator", Role = "coordinator" }
  depends_on = [aws_iam_role_policy_attachment.ssm, aws_iam_role_policy.results]
}

resource "aws_instance" "near" {
  ami                                  = data.aws_ssm_parameter.al2023_ami.value
  instance_type                        = var.instance_type
  subnet_id                            = var.near_subnet_id
  placement_group                      = aws_placement_group.near.name
  associate_public_ip_address          = true
  vpc_security_group_ids               = [aws_security_group.experiment.id]
  iam_instance_profile                 = aws_iam_instance_profile.experiment.name
  instance_initiated_shutdown_behavior = "terminate"
  user_data                            = local.node_user_data.near
  user_data_replace_on_change          = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 16
    encrypted             = true
    delete_on_termination = true
    tags                  = local.common_tags
  }

  tags       = { Name = "dai-${var.run_id}-near", Role = "near-worker" }
  depends_on = [aws_iam_role_policy_attachment.ssm, aws_iam_role_policy.results]
}

resource "aws_instance" "far" {
  ami                                  = data.aws_ssm_parameter.al2023_ami.value
  instance_type                        = var.instance_type
  subnet_id                            = var.far_subnet_id
  associate_public_ip_address          = true
  vpc_security_group_ids               = [aws_security_group.experiment.id]
  iam_instance_profile                 = aws_iam_instance_profile.experiment.name
  instance_initiated_shutdown_behavior = "terminate"
  user_data                            = local.node_user_data.far
  user_data_replace_on_change          = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 16
    encrypted             = true
    delete_on_termination = true
    tags                  = local.common_tags
  }

  tags       = { Name = "dai-${var.run_id}-far", Role = "far-worker" }
  depends_on = [aws_iam_role_policy_attachment.ssm, aws_iam_role_policy.results]
}
