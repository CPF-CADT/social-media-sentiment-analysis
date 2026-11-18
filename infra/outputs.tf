output "instance_id" {
  description = "Sentiment pipeline EC2 instance ID."
  value       = aws_instance.sentiment_pipeline.id
}

output "public_ip" {
  description = "Public IP address for SSH access."
  value       = aws_instance.sentiment_pipeline.public_ip
}

output "availability_zone" {
  description = "Availability zone shared by the EC2 instance and EBS volume."
  value       = aws_instance.sentiment_pipeline.availability_zone
}

output "data_volume_id" {
  description = "Persistent EBS volume ID."
  value       = aws_ebs_volume.data.id
}

output "data_mount_point" {
  description = "Persistent data mount point on the instance."
  value       = var.mount_point
}
