output "run_id" {
  value = var.run_id
}

output "coordinator_instance_id" {
  value = aws_instance.coordinator.id
}

output "near_instance_id" {
  value = aws_instance.near.id
}

output "far_instance_id" {
  value = aws_instance.far.id
}

output "near_private_ip" {
  value = aws_instance.near.private_ip
}

output "far_private_ip" {
  value = aws_instance.far.private_ip
}

output "near_availability_zone" {
  value = aws_instance.near.availability_zone
}

output "far_availability_zone" {
  value = aws_instance.far.availability_zone
}

output "result_bucket" {
  value = aws_s3_bucket.results.id
}
