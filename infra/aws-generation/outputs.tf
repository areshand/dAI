output "instance_id" {
  value = aws_instance.generation.id
}

output "result_bucket" {
  value = aws_s3_bucket.results.id
}

output "availability_zone" {
  value = aws_instance.generation.availability_zone
}
