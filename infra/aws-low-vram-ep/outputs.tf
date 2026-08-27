output "instance_ids" {
  value = aws_instance.worker[*].id
}

output "private_ips" {
  value = aws_instance.worker[*].private_ip
}

output "availability_zone" {
  value = aws_instance.worker[0].availability_zone
}

output "result_bucket" {
  value = aws_s3_bucket.results.id
}
