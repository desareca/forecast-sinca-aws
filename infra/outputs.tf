output "data_bucket_arn" {
  value = aws_s3_bucket.data.arn
}

output "mlflow_bucket_arn" {
  value = aws_s3_bucket.mlflow.arn
}

output "sagemaker_domain_id" {
  value = aws_sagemaker_domain.studio.id
}

output "sagemaker_space_name" {
  value = aws_sagemaker_space.dev.space_name
}

output "dev_role_arn" {
  value = aws_iam_role.dev.arn
}

output "training_role_arn" {
  value = aws_iam_role.training.arn
}
